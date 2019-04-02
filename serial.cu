#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <iostream>
#include <sys/time.h>
#include <stack>

#define G 6.67408E-11	//Gravitational constant 
#define lvl 9		//depth of quad tree till which we'll divide plane

using namespace std;

struct vect		//Structure for 2D coordinate
{
	float x; // X coordinate
	float y; // Y coordinate
};

struct node		//Structure for each node of the quad tree
{
	vect body; //centre of mass of bodies in current node

	float mass; //total mass of bodies in current node
	int child[4]; //children indices in nodes array
	int l,r; //index limit in body array of bodies in current node

	vect min, max; //min and max X and Y coordinates of bodies belonging to current node
};

//Function calculate Gravitational force between two body
vect gravity (vect a, vect b, float m1, float m2)
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

//This function will construct particular level of the tree.
//Each node will be divided further into four new nodes and bodies in the array will be swapped so that bodies belonging to same node remain together in the array

void construct(vect *body, float *mass, node *nodes, int level, int tot)
{
	int index;
	int total = 1<<(2*level);    //total nodes in current level
	int offset=((1<<(2*level))-1)/3;   //total nodes in tree upto previous level
	int off=offset+total;    //total nodes upto current level in tree

	for (int m=0; m<total; m++)
	{
		int tid=m*4;

		index=m+offset; //actual index in nodes array

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
				for (int j=0;j<4;j++)	child[i].child[j]=-1;
				child[i].min.x=child[i].min.y=FLT_MAX, child[i].max.x=child[i].max.y=FLT_MIN;
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
	}
}

//This function calculates force on bodies
void calculate(vect *body, float *mass, node *nodes, vect *force, int n, float theta)
{
	int l=((1<<(2*(lvl-1)))-1)/3; //total nodes in tree upto max depth

	for (int index=0;index<n;index++)
	{
		stack <int> st;

		st.push(0);

		vect bd=body[index];

		while (!st.empty()) //for each body do DFS until reached leaf
		{
			int t=st.top();
			st.pop();

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
				
					st.push(temp);
				}
			}
		}
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
	{	//Here we're going to take random inputs
		body[i].x=rand()%1000000;
		body[i].y=rand()%1000000;
		mass[i]=rand()%1000000;

		m+=mass[i], x+=body[i].x, y+=body[i].y;
	}	

	x/=n, y/=n; //centre of mass of the system

	vect force[n];

	vect mn, mx;

	mn.x=mn.y=FLT_MAX, mx.x=mx.y=FLT_MIN;

	struct timeval start, stop;

	gettimeofday(&start, NULL);

	for (int i=0;i<n;i++)
	{
		force[i].x=force[i].y=0;

		mn.x=min(mn.x, body[i].x);
		mx.x=max(mx.x, body[i].x);

		mn.y=min(mn.y, body[i].y);
		mx.y=max(mx.y, body[i].y);
	}

	gettimeofday(&stop, NULL);

	long seconds=(stop.tv_sec-start.tv_sec);

	float t=seconds*1000+float(stop.tv_usec-start.tv_usec)/1000;

	vect gmin=mn, gmax=mx;

	printf("%f %f\n %f %f\n", gmin.x, gmin.y, gmax.x, gmax.y);

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

	for (int i=0;i<lvl-1;i++) //creation of tree level by level
	{
		gettimeofday(&start, NULL);
		construct(body, mass, h_nodes, i, tot);
		gettimeofday(&stop, NULL);

		t+=((stop.tv_sec-start.tv_sec)*1000+float(stop.tv_usec-start.tv_usec)/1000);
	}
	
	float theta;

	printf("theta : ");
	scanf("%f", &theta);

	gettimeofday(&start, NULL);
	calculate(body, mass, h_nodes, force, n, theta);
	gettimeofday(&stop, NULL);

	t+=((stop.tv_sec-start.tv_sec)*1000+float(stop.tv_usec-start.tv_usec)/1000);
//t shows total number of time this serial code take
	x=0, y=0;

	for(int i=0;i<n;i++)
	{
		printf("%d : x %f y %f m %f : %.15f %.15f\n", i, body[i].x, body[i].y, mass[i], force[i].x, force[i].y);
		x+=force[i].x, y+=force[i].y;
	}

//	printf("%f %f\n", x, y);

	printf("cpu time : %f\n", t);
}

