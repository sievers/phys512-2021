#include <stdio.h>
#include <math.h>
//you can compile like so:
//gcc -fopenmp -o integrate_openmp integrate_openmp.c -O3 -lm

double integrate (double x0, double x1, double dx)
{
  int n=(x1-x0)/dx;
  if ((n&1)==0)
    n++;
  printf("n is %d\n",n);
  dx=(x1-x0)/(n-1);
  double tot=exp(x0)+exp(x1);
  //start a parallel region, every thread shares copies of variables
#pragma omp parallel shared(x0,x1,n,tot,dx) default(none)
  {
    //anything declared here is now going to be thread-private
    //we'll use mytot to sum up our private copy
    double mytot=0;
    //handy-dandy utility that splits up a loop amongst threads
#pragma omp for
    for (int i=1;i<n-1;i++) {
      int fac=2+2*(i&1);
      mytot=mytot+fac*exp(x0+i*dx);
    }
    //only let one thread at a time in here, avoids race condition
    //when reducing 
#pragma omp critical
    tot+=mytot;
  }
  return tot*dx/3;
}


/*--------------------------------------------------------------------------------*/
double integrate_parfor (double x0, double x1, double dx)
{
  int n=(x1-x0)/dx;
  if ((n&1)==0)
    n++;
  printf("n is %d\n",n);
  dx=(x1-x0)/(n-1);

  double tot=0;
  //start a parallel region and loop at the same time.
  //we'll accumulate into tot, and reduction tells the 
  //compiler we want that.
#pragma omp parallel for shared(x0,x1,n,dx) reduction(+:tot) default(none)
    for (int i=1;i<n-1;i++) {
      int fac=2+2*(i&1);
      tot+=fac*exp(x0+i*dx);
    }
  tot+=exp(x0)+exp(x1);
  return tot*dx/3;
}

/*================================================================================*/

int main(int argc, char *argv[])
{
  double x0=0;
  double x1=1;
  double dx=1e-4;
  double ans=integrate(x0,x1,dx);
  double ans2=integrate_parfor(x0,x1,dx);
  printf("Integrate values are %g %g\n",ans,ans2);
}
