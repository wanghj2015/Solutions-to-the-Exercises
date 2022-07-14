/**
 *
 * A test program for 1d flux-corrected transport
 *
 */


#include <cmath>
#include <string>
#include <fstream>
#include <iostream>

#include "flux_corrected_transport.h"

using namespace std;


int main()
{

const int    K = 1000;  // number of elements/cells
const double dx = 1.0;  // cell size

const double rho_min = 1.0e-6;  // set the density minimum
                                

double rho[K];
double u[K];
double rhou[K];

double rho_new[K];
double rhou_new[K];


// initialize the model

for (int i = 0; i < K; ++i)
{
   //u[i] = 10.0;
   u[i] = 10.0*sin(i*dx*2*M_PI/1000);

   if (250 <= i && i <= 449)
      rho[i] = 25.0;
   else
      rho[i] = 1.0e-5;

   rhou[i] = rho[i] * u[i];
}


// save initial conditions

// open the output file
std::ofstream output_density("density.dat", std::ios::binary | std::ios::out);
if (!output_density.is_open()) { // check for successful opening
    cout << "Output file could not be opened! Terminating!" << endl;
    return 1;
    }

output_density.write(reinterpret_cast<const char*>(&rho), std::streamsize(K*sizeof(double)));


std::ofstream output_velocity("velocity.dat", std::ios::binary | std::ios::out);

output_velocity.write(reinterpret_cast<const char*>(&u), std::streamsize(K*sizeof(double)));


// marching in time

double time = 0.0;
double FinalTime = 15.0;

const double CFL = 0.5;

while (time < FinalTime) 
{

// determine dt 
double umax = abs(*max_element(begin(u), end(u)));
double dt = CFL * dx / umax;

if (time + dt > FinalTime)
   dt = FinalTime - time;


cout << "  The maximum velocity [m/s]: " << umax << endl;
cout << "  Time step size in seconds : " << dt << endl;


// transport for density
flux_corr_method(rho, u, K, dt, dx, rho_new);


// transport for momentum
//flux_corr_method(rhou, u, K, dt, dx, rhou_new);


// update rho and get u from rhou
for (int i = 0; i < K; ++i)
{
   //rho[i] = rho_new[i];
   rho[i] = max(rho_new[i], rho_min);
}


// transport for momentum
flux_corr_method(rhou, u, K, dt, dx, rhou_new);

// update rhou and get u from rhou
for (int i = 0; i < K; ++i)
{
   rhou[i] = rhou_new[i];
   u[i] = rhou[i] / rho[i];
}


time += dt;         // increment time


// output density
output_density.write(reinterpret_cast<const char*>(&rho), std::streamsize(K*sizeof(double)));

output_velocity.write(reinterpret_cast<const char*>(&u), std::streamsize(K*sizeof(double)));

}

// close file
output_density.close();
output_velocity.close();

return 0;

}


