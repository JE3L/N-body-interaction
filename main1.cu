#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <iostream>
#include <sys/time.h>

#define G 6.67408E-11 //Gravitational constant 

#define lvl 9		//depth of quad tree till which we'll divide plane

using namespace std;

struct vect		//Structure for 2D coordinate
{
	float x; //X coordinate
	float y; //Y coordinate
};

struct node		//Structure for each node of the quad tree

{
	vect body; //centre of mass of bodies in current node
	float mass; //total mass of bodies in current node
	int child[4]; //children indices in nodes array
	int l,r; //index limit in body array of bodies in current node
	vect min, max; //min and max X and Y coordinates of bodies belonging to current node
};

//Function to calculate Gravitational force between two bodies
__device__ vect gravity (vect a, vect b, float m1, float m2)
{
	float res=G*m1*m2;

	float r=(a.y-b.y)*(a.y-b.y)+(a.x-b.x)*(a.x-b.x);

	if (r>0)	res/=r;

	vect vec;

	vec.y=a.y-b.y;
	vec.x=a.x-b.x;

	r=sqrt(r);

	if (r>0)	vec.y/=r, vec.x/=r;

	vec.y*=res;
	vec.x*=res;

	return vec;
}

//part1 for kernel1 to find min-max among the n bodies 
//Will find min and max of X and Y coordinates for each thread lock
//Uses reduction technique
__global__ void findMinMax(vect * body, vect * min, vect * max, int n)
{
	__shared__ vect min_cache[32];
	__shared__ vect max_cache[32];

	int index=threadIdx.x+blockDim.x*blockIdx.x;

	float xmin=FLT_MAX, ymin=FLT_MAX;
	float xmax=FLT_MIN, ymax=FLT_MIN;

	while (index<n) //takes care if total number greater than total threads in kernel
	{
		xmin=fmin(xmin, body[index].x);
		ymin=fmin(ymin, body[index].y);
		xmax=fmax(xmax, body[index].x);
		ymax=fmax(ymax, body[index].y);

		index+=(blockDim.x*gridDim.x); //incrementing index by total number of threads in kernel, to take care if total number more than total threads in kernel
	}

	int tid=threadIdx.x;

	min_cache[tid].x=xmin;
	min_cache[tid].y=ymin;

	max_cache[tid].x=xmax;
	max_cache[tid].y=ymax;

	int active=blockDim.x>>1;

	do
	{
		__syncthreads();

		if (tid<active) //reduction
		{
			min_cache[tid].x=fmin(min_cache[tid].x, min_cache[tid+active].x);
			min_cache[tid].y=fmin(min_cache[tid].y, min_cache[tid+active].y);

			max_cache[tid].x=fmax(max_cache[tid].x, max_cache[tid+active].x);
			max_cache[tid].y=fmax(max_cache[tid].y, max_cache[tid+active].y);
		}

		active>>=1;
	}while (active>0);

	if (tid==0)	min[blockIdx.x]=min_cache[0], max[blockIdx.x]=max_cache[0];
}

//part2 for kernel1 to find min-max among the n bodies 
//Will find global min and max of X and Y coordinates from local min and max of above kernel

__global__ void findMMinMax(vect * mmin, vect *mmax, vect * min, vect * max, int n)
{
	__shared__ vect min_cache[32];
	__shared__ vect max_cache[32];

	int index=threadIdx.x+blockDim.x*blockIdx.x;

	float xmin=FLT_MAX, ymin=FLT_MAX;
	float xmax=FLT_MIN, ymax=FLT_MIN;

	while (index<n) //takes care if total number greater than total threads in kernel
	{
		xmin=fmin(xmin, min[index].x);
		ymin=fmin(ymin, min[index].y);
		xmax=fmax(xmax, max[index].x);
		ymax=fmax(ymax, max[index].y);

		index+=(blockDim.x*gridDim.x);
	}

	int tid=threadIdx.x;

	min_cache[tid].x=xmin;
	min_cache[tid].y=ymin;

	max_cache[tid].x=xmax;
	max_cache[tid].y=ymax;

	int active=blockDim.x>>1;

	do
	{
		__syncthreads();

		if (tid<active) //reduction
		{
			min_cache[tid].x=fmin(min_cache[tid].x, min_cache[tid+active].x);
			min_cache[tid].y=fmin(min_cache[tid].y, min_cache[tid+active].y);

			max_cache[tid].x=fmax(max_cache[tid].x, max_cache[tid+active].x);
			max_cache[tid].y=fmax(max_cache[tid].y, max_cache[tid+active].y);
		}

		active>>=1;
	}while (active>0);

	if (tid==0)	mmin[blockIdx.x]=min_cache[0], mmax[blockIdx.x]=max_cache[0];
}

//This function will construct particular level of the tree.
//Each node will be divided further into four new nodes and bodies in the array will be swapped so that bodies belonging to same node remain together in the array

//This will work as kernel 2
__global__ void construct(vect *body, float *mass, node *nodes, int level, int tot)
{
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	
	int tid=index*4;

	int total = 1<<(2*level); //total nodes in current level

	int offset=((1<<(2*level))-1)/3; //total nodes in tree upto previous level
	int off=offset+total; //total nodes in tree upto current level

	while (index<total) //'while' loop will take care if total number more than total threads in kernel
	{
		index+=offset; //actual index in nodes array

		node nd=nodes[index];

		if (nodes[index].l<=nodes[index].r)
		{
			float xl=nd.min.x, xr=nd.max.x;
			float yl=nd.min.y, yr=nd.max.y;

			float xmid=xl+(xr-xl)/2;
			float ymid=yl+(yr-yl)/2;

			float l=nd.l, r=nd.r;

			node child[4];

			for (int i=0;i<4;i++)	
			{
				child[i].min.x=child[i].min.y=FLT_MAX, child[i].max.x=child[i].max.y=FLT_MIN;
					
				for (int j=0;j<4;j++)	child[i].child[j]=-1;
			}

			int i=l-1;

			float m=0, x=0, y=0, mm=0, xx=0, yy=0;
			
			for (int j=l;j<=r;j++) //swapping of bodies belonging to current node based on x-coordinates creating two children

			{
				if (body[j].x<=xmid)
				{
					i++;
					
					vect temp=body[i];
					body[i]=body[j];
					body[j]=temp;

					float t=mass[i];
					mass[i]=mass[j];
					mass[j]=t;
				}

			}

			child[2].l=l, child[2].r=i;
			child[3].l=i+1, child[3].r=r;

			for (int k=2;k<=3;k++)
			{
				m=mm=x=xx=y=yy=0;

				l=child[k].l, r=child[k].r;

				i=l-1;

				int cnt=0;

				for (int j=l;j<=r;j++) //swapping of bodies in two children created previously based on y-coordinates, each creating two new children

				{
					x+=body[j].x;
					y+=body[j].y;
					m+=mass[j];

					if (body[j].y<=ymid)
					{
						xx+=body[j].x, yy+=body[j].y, mm+=mass[j];
						cnt++;
						i++;
						
						vect temp=body[i];
						body[i]=body[j];
						body[j]=temp;

						float t=mass[i];
						mass[i]=mass[j];
						mass[j]=t;
					}
				}
				
				if(cnt>0)	child[k].mass=mm, child[k].body.x=xx/cnt, child[k].body.y=yy/cnt;

				child[k].l=l, child[k].r=i;

				mm=m-mm, xx=x-xx, yy=y-yy, cnt=r-l+1-cnt;
				
				if(cnt>0)	child[k-2].mass=mm, child[k-2].body.x=xx/cnt, child[k-2].body.y=yy/cnt;

				child[k-2].l=i+1, child[k-2].r=r;
			}			

			for (int i=0;i<4;i++)	
			{
				if (i%2)	child[i].min.x=xmid, child[i].max.x=xr;
				else	child[i].min.x=xl, child[i].max.x=xmid;
				
				if (i<2)	child[i].min.y=ymid, child[i].max.y=yr;
				else	child[i].min.y=yl, child[i].max.y=ymid;

				if (off+tid+i<tot)	nodes[off+tid+i]=child[i];
				nd.child[i]=off+tid+i;
			}	
		}
		else
		{
			for (int i=0;i<4;i++)
			{
				if (off+tid+i<tot)
				{	
					nodes[off+tid+i].l=0;
					nodes[off+tid+i].r=-1;
				}

				nd.child[i]=off+tid+i;
			}
		}

		nodes[index]=nd;

		index-=offset;
		
		index+=(blockDim.x*gridDim.x); //will take care if total number more than total threads by incrementing index by total threads.
	}
}

//This is kernel 3
//This function calculates force on bodies

__global__ void calculate(vect *body, float *mass, node *nodes, vect *force, int n, float theta)
{
	int index=blockDim.x*blockIdx.x+threadIdx.x;

	int l=((1<<(2*(lvl-1)))-1)/3; //total nodes in tree upto max depth


	while (index<n) //'while' loop takes care if total number more than total threads in kernel
	{
		int st[4*(lvl)]; //using array as stack
		int curr=0; //variable showing current top index

		st[curr]=0;

		vect bd=body[index];

		while (curr>=0) //for each body do DFS until reached leaf

		{
			int t=st[curr];
			curr--;

			node nd=nodes[t];

			float s=fmax(nd.max.x-nd.min.x, nd.max.y-nd.min.y);

			float x=bd.x-nd.body.x, y=bd.y-nd.body.y;

			float dist=sqrt(x*x+y*y);

			float val=FLT_MAX;

			if (dist>0)	val=s/dist;

			if (val<theta)	//Barnes-Hutt approximation
			{
				vect frc=gravity(nd.body, bd, nd.mass, mass[index]);

				force[index].x+=frc.x;
				force[index].y+=frc.y;
			}
			else
			{
				if (t>=l) //if reached leaf
				{
					vect frc=gravity(nd.body, bd, nd.mass, mass[index]);

					force[index].x+=frc.x;
					force[index].y+=frc.y;

					continue;
				}

				for (int i=0;i<4;i++)
				{
					int temp=nd.child[i];

					if (temp==-1 || nodes[temp].l>nodes[temp].r)	continue;
					
					st[++curr]=temp;
				}
			}
		}

		index+=(blockDim.x*gridDim.x); //will take care if total number more than total threads by incrementing index by total number of threads in kernel
	}
}

float maxx(float a, float b)
{
	return (a<b?b:a);
}

float minn(float a, float b)
{
	return (a<b?a:b);
}

int main()
{
	int n;
	printf("n : ");
	scanf("%d", &n);

	vect body[n];

	float mass[n];

	float m=0, x=0, y=0;

	for (int i=0;i<n;i++)	
	{
		body[i].x=rand()%1000000;
		body[i].y=rand()%1000000;
		mass[i]=rand()%1000000;

		m+=mass[i], x+=body[i].x, y+=body[i].y;
	}	

	x/=n, y/=n; //centre of mass of the whole system

	vect force[n];
	vect *min;
	vect *max;

	for (int i=0;i<n;i++)	force[i].x=force[i].y=0;

	vect *dforce;
	vect *dbody;
	float *dmass;

	int s=sizeof(vect)*n;
	int sz=sizeof(float)*n;

	cudaMalloc(&dbody, s);
	cudaMalloc(&dmass, sz);

	cudaMalloc(&dforce, s);
	cudaMemset(dforce, 0, s);

	cudaMalloc(&min, s);
	cudaMalloc(&max, s);

	cudaMemcpy(dbody, body, s, cudaMemcpyHostToDevice);
	cudaMemcpy(dmass, mass, sz, cudaMemcpyHostToDevice);

	int val=32;

	int block = val;

	int grid = val;

	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);

	//This is kernel 1 which devided into 2 parts
	//bassically it find minimum and maximum from
	//the n bodies
	findMinMax<<<grid, block>>>(dbody, min, max, n);
	
	findMMinMax<<<1, block>>>(min, max, min, max, val);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//milliseconds find the total kernal time in GPU
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	vect gmin;
	vect gmax;

	cudaMemcpy(&gmin, min, sizeof(vect), cudaMemcpyDeviceToHost);
	cudaMemcpy(&gmax, max, sizeof(vect), cudaMemcpyDeviceToHost);

	int curr=1;

	int tot=1<<(2*lvl);
	tot=(tot-1)/3;

	node h_nodes[tot];

	for (int i=0;i<tot;i++)
	{
		for (int j=0;j<4;j++)	h_nodes[i].child[j]=-1;
	}

	vect temp;
	temp.x=x, temp.y=y;

	h_nodes[0].body=temp, h_nodes[0].mass=m, h_nodes[0].l=0, h_nodes[0].r=n-1, h_nodes[0].min=gmin, h_nodes[0].max=gmax;

	node *d_nodes;
	cudaMalloc(&d_nodes, sizeof(node)*tot);
	cudaMemcpy(d_nodes, h_nodes, sizeof(node)*tot, cudaMemcpyHostToDevice);

	float t;

	for (int i=0;i<lvl-1;i++) //creation of tree level by level. Each thread is assigned a node in current level.
	{
		block = 1024;
		grid=ceil((1.0*curr)/block);

		grid = minn(grid, 20);

		cudaEventRecord(start);
		construct<<<grid, block>>>(dbody, dmass, d_nodes, i, tot);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&t, start, stop);

		milliseconds+=t;

		curr*=4;
	}

	cudaMemcpy(h_nodes, d_nodes, sizeof(node)*tot, cudaMemcpyDeviceToHost);

	float theta;

	printf("theta : ");
	scanf("%f", &theta);

	grid=minn(20, ceil((1.0*n)/block));

	printf("%d\n", grid);

	cudaEventRecord(start);
	calculate<<<grid, block>>>(dbody, dmass, d_nodes, dforce, n, theta);  //Each thread is assigned a body
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&t, start, stop);

	milliseconds+=t;

	cudaMemcpy(force, dforce, s, cudaMemcpyDeviceToHost);
	cudaMemcpy(body, dbody, s, cudaMemcpyDeviceToHost);
	cudaMemcpy(mass, dmass, sz, cudaMemcpyDeviceToHost);

	x=0, y=0;

	for(int i=0;i<n;i++)
	{
		printf("force %d : x %f y %f m %f : %.15f %.15f\n", i, body[i].x, body[i].y, mass[i], force[i].x, force[i].y);
		x+=force[i].x, y+=force[i].y;
	}

	printf("gpu time : %f\n", milliseconds);

	printf("x %f y %f\n", x, y);
}

