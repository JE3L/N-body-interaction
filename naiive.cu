//This is simple naive programme(brute force)
//which runs in O(N^2)
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <sys/time.h>

#define G 6.67408E-11

struct vector
{
	float x, y;
};

//This function calculate the gravitational force
//between two bodies or particals
vector gravity (vector a, vector b, float m1, float m2)
{
	float res=G*m1*m2;

	float r=(a.y-b.y)*(a.y-b.y)+(a.x-b.x)*(a.x-b.x);

	if (r>0)	res/=r;

	vector vec;

	vec.y=a.y-b.y;
	vec.x=a.x-b.x;

	r=sqrt(r);

	if (r>0)
	{
		vec.y/=r;
		vec.x/=r;
	}

	vec.y*=res;
	vec.x*=res;

	return vec;
}

int main()
{
	int n;
	scanf("%d", &n);

	vector body[n];

	float mass[n];

	for (int i=0;i<n;i++)
	{
		body[i].x=rand()%1000000;
		body[i].y=rand()%1000000;
		mass[i]=rand()%1000000;

//		scanf("%lf %lf %lf", &body[i].x, &body[i].y, &mass[i]);
	}

	vector force[n];

	for (int i=0;i<n;i++)	force[i].x=force[i].y=0;

	struct timeval start, stop;

	gettimeofday(&start, NULL);

	float x=0, y=0;

	for (int i=0;i<n;i++)
	{
		for (int j=i+1;j<n;j++)
		{
			vector temp=gravity(body[i], body[j], mass[i], mass[j]);

			force[i].x+=temp.x;
			force[i].y+=temp.y;

			force[j].x-=temp.x;
			force[j].y-=temp.y;
		}

		printf("%d : %f %f %f :  %.15f %.15f\n", i, body[i].x, body[i].y, mass[i], force[i].x, force[i].y);

		x+=force[i].x, y+=force[i].y;
	}

	gettimeofday(&stop, NULL);

	float t=((stop.tv_sec-start.tv_sec)*1000+float(stop.tv_usec-start.tv_usec)/1000);

	printf("naiive : %f\n", t);
	printf("%f %f\n", x, y);
}
