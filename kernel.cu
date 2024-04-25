#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>


static uint8_t* buffer=0;
static double* vecl=0;
static uint8_t* stars = 0;

const static int starsize = 100;


__device__ void tormat(double phi, double theta, double* mat)
{
	mat[0] = cos(theta) * sin(phi);
	mat[3] = cos(theta) * cos(phi);
	mat[6] = sin(theta);

	mat[2] = cos(phi);
	mat[5] = -sin(phi);
	mat[8] = 0;

	mat[1] = -sin(theta) * sin(phi);
	mat[4] = -sin(theta) * cos(phi);
	mat[7] = cos(theta);
}

__device__ double matdet(double* m)
{
	return m[0] * (m[4]*m[8]-m[5]*m[7]) - m[1] * (m[3]*m[8]-m[5]*m[6]) + m[2] * (m[3]*m[7]-m[4]*m[6]);
}

__device__ void matinv(double* m, double* res)
{
	res[0] = m[4] * m[8] - m[5] * m[7];
	res[1] = m[2] * m[7] - m[1] * m[8];
	res[2] = m[1] * m[5] - m[2] * m[4];
	res[3] = m[5] * m[6] - m[3] * m[8];
	res[4] = m[0] * m[8] - m[2] * m[6];
	res[6] = m[3] * m[7] - m[4] * m[6];
	res[5] = m[2] * m[3] - m[0] * m[5];
	res[7] = m[1] * m[6] - m[0] * m[7];
	res[8] = m[0] * m[4] - m[1] * m[3];
}

__device__ void matmult(double* m1, double* m2, double* res)
{
	res[0] = m1[0] * m2[0] + m1[1] * m2[3] + m1[2] * m2[6];
	res[1] = m1[0] * m2[1] + m1[1] * m2[4] + m1[2] * m2[7];
	res[2] = m1[0] * m2[2] + m1[1] * m2[5] + m1[2] * m2[8];
	res[3] = m1[3] * m2[0] + m1[4] * m2[3] + m1[5] * m2[6];
	res[4] = m1[3] * m2[1] + m1[4] * m2[4] + m1[5] * m2[7];
	res[5] = m1[3] * m2[2] + m1[4] * m2[5] + m1[5] * m2[8];
	res[6] = m1[6] * m2[0] + m1[7] * m2[3] + m1[8] * m2[6];
	res[7] = m1[6] * m2[1] + m1[7] * m2[4] + m1[8] * m2[7];
	res[8] = m1[6] * m2[2] + m1[7] * m2[5] + m1[8] * m2[8];
}

__device__ void matact(double* m, double vecn0, double vecn1, double vecn2, double* nvecn)
{
	nvecn[0] = m[0] * vecn0 + m[1] * vecn1 + m[2] * vecn2;
	nvecn[1] = m[3] * vecn0 + m[4] * vecn1 + m[5] * vecn2;
	nvecn[2] = m[6] * vecn0 + m[7] * vecn1 + m[8] * vecn2;
}

__device__ void matflip(double* m, double* res)
{
	res[0] = m[6];
	res[1] = m[7];
	res[2] = m[8];
	res[3] = m[3];
	res[4] = m[4];
	res[5] = m[5];
	res[6] = -m[0];
	res[7] = -m[1];
	res[8] = -m[2];
}

__device__ void matflip2(double* m, double* res)
{
	res[0] = m[2];
	res[1] = m[1];
	res[2] = -m[0];
	res[3] = m[5];
	res[4] = m[4];
	res[5] = -m[3];
	res[6] = m[8];
	res[7] = m[7];
	res[8] = -m[6];
}

__device__ double solvequartic(double a0, double b0, double c0, double d0, double e0)
{
	double tmp;
	double tmin = 65536.0;
	double sint,s;
	double r1, qds, rootint;

	double b = b0 / a0;
	double c = c0 / a0;
	double d = d0 / a0;
	double e = e0 / a0;

	double c2 = c * c;
	double bd = b * d;
	double c3 = c2 * c;
	double bcd = bd * c;
	double b2 = b * b;
	double b2e = b2 * e;
	double d2 = d * d;
	double ce = c * e;
	double bc = b * c;
	double b3 = b2 * b;
	double mbd4 = (-0.25) * b;

	double t0 = c2 - 3.0 * bd + 12.0 * e;
	double t1 = 2.0 * c3 - 9.0 * bcd + 27.0 * b2e + 27.0 * d2 - 72.0 * ce;
	double p = (8.0 * c - 3.0 * b2) / 8.0;
	double q = (b3 - 4.0 * bc + 8.0 * d) / 8.0;

	double disc = t1 * t1 - 4.0 * t0 * t0 * t0;
	
	if (disc < 0)
	{
		double st0 = sqrt(t0);
		double phi = (acos(t1 / (2.0 * t0 * st0))) / 3.0;
		sint = (-2.0 / 3.0) * p + (2.0 / 3.0) * st0 * cos(phi);
	}
	else
	{
		double bigq = cbrt((t1 + sqrt(disc)) * 0.5);
		sint = (-2.0 / 3.0) * p + (1.0 / 3.0) * (bigq + t0 / bigq);
	}	

	s = sqrt(sint) * 0.5;
	
	rootint = (sint + 2.0 * p) * (-1.0);
	qds = q / s;

	r1 = rootint + qds;

	if (r1 > 0)
	{
		r1 = 0.5 * sqrt(r1);
		tmp = mbd4 - s;

		if (tmp + r1 > 0.0000001 && tmp + r1 < tmin) tmin = tmp + r1;
		if (tmp - r1 > 0.0000001 && tmp - r1 < tmin) tmin = tmp - r1;
	}

	r1 = rootint - qds;

	if (r1 > 0)
	{
		r1 = 0.5 * sqrt(r1);
		tmp = mbd4 + s;

		if (tmp + r1 > 0.0000001 && tmp + r1 < tmin) tmin = tmp + r1;
		if (tmp - r1 > 0.0000001 && tmp - r1 < tmin) tmin = tmp - r1;
	}

	return tmin;
}

__device__ double toruscoll(double a, double b, double c, double d, double e, double f, double m, double n)
{
	double t4, t3, t2, t1, t0;

	double a2 = a * a;
	double b2 = b * b;
	double c2 = c * c;
	double d2 = d * d;
	double e2 = e * e;
	double f2 = f * f;

	double ab = a * b;
	double cd = c * d;
	double ef = e * f;
	double abc = ab * c;

	double sum1 = a2 + c2 + e2;
	double sum2 = ab + cd;
	double sum3 = sum2 + ef;
	double sum4 = m + n;
	double sum5 = b2 + d2 + f2;
	double sum6 = m - n;
	double sum7 = ab + ef;

	t0 = sum5 * sum5 + sum6 * sum6;
	t0 += (-2.0) * (sum5 * sum4 - 2.0 * f2 * n);

	t1 = (b2 + d2 + f2) * sum3;
	t1 -= sum3 * sum4;
	t1 += 2.0 * ef * n;
	t1 *= 4.0;

	t2 = d * (d * (sum1 + 2.0 * c2) + 4.0 * c * sum7) + b * (b * (sum1 + 2.0 * a2) + 4.0 * a * ef) + f2 * (sum1 + 2.0 * e2);
	t2 -= sum1 * sum4;
	t2 += 2.0 * e2 * n;
	t2 *= 2.0;

	t3 = 4.0 * sum1 * sum3;

	t4 = sum1 * sum1;

	return solvequartic(t4, t3, t2, t1, t0);
}

__global__ void bufferinit(uint8_t* buffer)
{
	buffer[4 * (blockIdx.x * blockDim.x + threadIdx.x) + 3] = 255;
}

__global__ void setstars(uint8_t* stars)
{
	int i;
	int tmp = blockIdx.x * blockDim.x + threadIdx.x;

	int rand = tmp;

	for (i = 0; i < 10; i++) rand = (60493 * rand + 11) % 115249;

	if ((rand) % 5 == 0)
	{
		stars[tmp] = 255 * rand / 115249;
	}
	else
	{
		stars[tmp] = 0;
	}
}

__global__ void addKernel(uint8_t* buffer, double* vecl, double pos0,double pos1, double pos2, double vec0, double vec1, double vec2, double addy0, double addy1, double addy2, double addz0, double addz1, double addz2, bool inside, double alpha, double beta, double bigr, double r, bool other, uint8_t* stars)
{
	int i;
	double vecn0, vecn1, vecn2;
	double roomsize = 10;
	double schecker = 1;

	double geoang;
	double tcont;
	double tmpr;
	double tmpx2;
	double xyvec;

	double inv[9]{};
	double nvecn[3]{};
	double npos[3]{};
	double vl;

	double torcoll;
	double tor0, tor1, tor2;
	double theta, phi;

	double tmin,tsol;
	int tmincoord;
	double tmpsign;
	double exitalpha;

	double rayon;
	double kappa;
	double exit;
	double leangle;

	double coll0, coll1;
	int checker;
	int ctmp0, ctmp1, ctmp2;

	double u, v;
	uint8_t uv;

	int tmp2;
	int tmp = blockIdx.x * blockDim.x + threadIdx.x;
	int tmpx = tmp % 1920;
	int tmpy = (tmp-tmpx) /1920;

	double mat1[9]{};

	vecn0 = vec0 + tmpx * addy0 + tmpy * addz0;
	vecn1 = vec1 + tmpx * addy1 + tmpy * addz1;
	vecn2 = vec2 + tmpx * addy2 + tmpy * addz2;

	vecn0 /= vecl[tmp];
	vecn1 /= vecl[tmp];
	vecn2 /= vecl[tmp];

	if (inside)
	{
		vl = sqrt(1.0 - vecn2 * vecn2);
		geoang = atan(vecn2 / vl);

		rayon = pos2 / cos(geoang);

		kappa = sin(geoang) * rayon; //(-1)*kappa

		exitalpha = sqrt(rayon * rayon - alpha * alpha);

		if(kappa>0 && beta<rayon)
		{
			exit = sqrt(rayon * rayon - beta * beta);
			
			pos0 += (kappa-2.0*exit+exitalpha) * vecn0 / vl;
			pos1 += (kappa-2.0*exit+exitalpha) * vecn1 / vl;

			other = !other;
		}
		else
		{
			pos0 += (exitalpha + kappa) * vecn0 / vl;
			pos1 += (exitalpha + kappa) * vecn1 / vl;
		}


		vecn0 = sqrt(1.0 - (exitalpha * exitalpha) / (rayon * rayon)) * vecn0 / vl;
		vecn1 = sqrt(1.0 - (exitalpha * exitalpha) / (rayon * rayon)) * vecn1 / vl;
		vecn2 = -exitalpha / rayon;

		pos0 /= bigr;
		pos1 /= r;

		npos[0] = sin(pos0) * (bigr + r * cos(pos1));
		npos[1] = cos(pos0) * (bigr + r * cos(pos1));
		npos[2] = r * sin(pos1);

		tormat(pos0, pos1, mat1);
		matflip2(mat1, inv);
		matact(inv, vecn0, vecn1, vecn2, nvecn);

		vecn0 = nvecn[0];
		vecn1 = nvecn[1];
		vecn2 = nvecn[2];

		pos0 = npos[0];
		pos1 = npos[1];
		pos2 = npos[2];
	}


	torcoll = toruscoll(vecn0, pos0, vecn1, pos1, vecn2, pos2, r*r, bigr*bigr);

	for (i = 0; i < 10; i++) {
		if (torcoll != 65536)
		{
			tor0 = pos0 + torcoll * vecn0;
			tor1 = pos1 + torcoll * vecn1;
			tor2 = pos2 + torcoll * vecn2;
			xyvec = sqrt(tor0 * tor0 + tor1 * tor1);

			theta = asin(tor2 / r);
			if (xyvec < bigr) theta = M_PI - theta;
			if (theta < 0) theta += 2.0 * M_PI;

			phi = acos(tor1 / xyvec);
			if (tor0 < 0) phi *= -1.0;
			if (phi < 0) phi += 2.0 * M_PI;

			tormat(phi, theta, mat1);
			matinv(mat1, inv);
			matflip(inv, mat1);
			matact(mat1, vecn0, vecn1, vecn2, nvecn);

			npos[0] = phi * (bigr);
			npos[1] = theta * r;


			vl = sqrt(1.0 - nvecn[2] * nvecn[2]);
			geoang = atan(nvecn[2] / vl);
			rayon = alpha / cos(geoang);
			kappa = sin(geoang) * rayon; //(-1)*kappa

			if (beta < rayon)
			{
				exitalpha = sqrt(rayon * rayon - alpha * alpha);
				exit = sqrt(rayon * rayon - beta * beta);

				npos[0] += (kappa - 2.0 * exit + exitalpha) * nvecn[0] / vl;
				npos[1] += (kappa - 2.0 * exit + exitalpha) * nvecn[1] / vl;

				nvecn[0] = sqrt(1.0 - (exitalpha * exitalpha) / (rayon * rayon)) * nvecn[0] / vl;
				nvecn[1] = sqrt(1.0 - (exitalpha * exitalpha) / (rayon * rayon)) * nvecn[1] / vl;
				nvecn[2] = -exitalpha / rayon;

				other = !other;
			}
			else
			{
				npos[0] += 2.0 * kappa * nvecn[0] / vl;
				npos[1] += 2.0 * kappa * nvecn[1] / vl;

				nvecn[2] *= -1.0;
			}
			

			npos[0] /= bigr;
			npos[1] /= r;


			pos0 = sin(npos[0]) * (bigr + r * cos(npos[1]));
			pos1 = cos(npos[0]) * (bigr + r * cos(npos[1]));
			pos2 = r * sin(npos[1]);

			tormat(npos[0], npos[1], mat1);
			matflip2(mat1, inv);
			matact(inv, nvecn[0], nvecn[1], nvecn[2], nvecn);

			vecn0 = nvecn[0];
			vecn1 = nvecn[1];
			vecn2 = nvecn[2];

			torcoll = toruscoll(vecn0, pos0, vecn1, pos1, vecn2, pos2, r * r, bigr * bigr);
		}
	}


	if (torcoll != 65536 || isnan(vecn0))
	{
		buffer[4 * tmp] = 0;
		buffer[4 * tmp + 1] = 0;
		buffer[4 * tmp + 2] = 0;
		return;
	}

	tcont = 0;
	

	if (other)
	{
		u = starsize * ((0.5 + atan2(vecn1, vecn0) / (2.0 * M_PI)));
		v = starsize * ((0.5 + asin(vecn2) / M_PI));

		tmp2 = (int)u + starsize * (int)v;
		//if (tmp2 < 0) uv = 0;
		//else uv = stars[tmp2];
		uv = stars[tmp2];

		if (uv % 3 == 0)
		{
			buffer[4 * tmp] = 0;
			buffer[4 * tmp + 1] = 0;
			buffer[4 * tmp + 2] = (uv * uv * uv * uv) / (255.0 * 255.0 * 255.0);
		}
		else
		{
			buffer[4 * tmp] = (uv*uv*uv) / ( 255.0*255.0);
			buffer[4 * tmp + 1] = (uv*uv*uv) / ( 255.0* 255.0);
			buffer[4 * tmp + 2] = (uv*uv*uv)/( 255.0* 255.0 );
		}
	}
	else
	{

		if (vecn0 < 0) tmpsign = -1;
		else tmpsign = 1;

		tmin = (tmpsign * roomsize - pos0) / vecn0;
		tmincoord = 0;

		if (vecn1 < 0) tmpsign = -1;
		else tmpsign = 1;
		tsol = (tmpsign * roomsize - pos1) / vecn1;
		if (tsol < tmin)
		{
			tmin = tsol;
			tmincoord = 1;
		}

		if (vecn2 < 0) tmpsign = -1;
		else tmpsign = 1;
		tsol = (tmpsign * roomsize - pos2) / vecn2;
		if (tsol < tmin)
		{
			tmin = tsol;
			tmincoord = 2;
		}

		if (tmincoord == 0)
		{
			coll0 = pos1 + tmin * vecn1;
			coll1 = pos2 + tmin * vecn2;
		}
		else if (tmincoord == 1)
		{
			coll0 = pos0 + tmin * vecn0;
			coll1 = pos2 + tmin * vecn2;
		}
		else
		{
			coll0 = pos0 + tmin * vecn0;
			coll1 = pos1 + tmin * vecn1;
		}

		checker = ((int)floor(coll0 * schecker)) % 2;
		checker += ((int)floor(coll1 * schecker)) % 2;
		if (checker < 0) checker += 2;
		checker %= 2;



		if (tmincoord == 2) {
			ctmp0 = 255;
			ctmp1 = 255;
			ctmp2 = 255;
		}
		else if (tmincoord == 0)
		{
			if (vecn0 < 0)
			{
				tmpx2 = (1.0 / 8.0) - (coll0 / (8.0 * roomsize));
				tmpr = fmod(tmpx2, 1.0 / 6.0);

				if (tmpx2 < 1.0 / 6.0)
				{
					ctmp0 = 255.0;
					ctmp1 = (int)(1530.0 * tmpr);
					ctmp2 = 0;
				}
				else if (tmpx2 < 1.0 / 3.0)
				{
					ctmp1 = 255.0;
					ctmp0 = (int)(255.0 - 1530.0 * tmpr);
					ctmp2 = 0;
				}
				else if (tmpx2 < 0.5)
				{
					ctmp1 = 255.0;
					ctmp2 = (int)(1530.0 * tmpr);
					ctmp0 = 0;
				}
				else if (tmpx2 < 2.0 / 3.0)
				{
					ctmp2 = 255.0;
					ctmp1 = (int)(255.0 - 1530.0 * tmpr);
					ctmp0 = 0;
				}
				else if (tmpx2 < 5.0 / 6.0)
				{
					ctmp2 = 255.0;
					ctmp0 = (int)(1530.0 * tmpr);
					ctmp1 = 0;
				}
				else
				{
					ctmp0 = 255.0;
					ctmp2 = (int)(255.0 - 1530.0 * tmpr);
					ctmp1 = 0;
				}
			}
			else
			{
				tmpx2 = (1.0 / 2.0) + (1.0 / 8.0) + coll0 / (8.0 * roomsize);

				tmpr = fmod(tmpx2, 1.0 / 6.0);

				if (tmpx2 < 1.0 / 6.0)
				{
					ctmp0 = 255.0;
					ctmp1 = (int)(1530.0 * tmpr);
					ctmp2 = 0;
				}
				else if (tmpx2 < 1.0 / 3.0)
				{
					ctmp1 = 255.0;
					ctmp0 = (int)(255.0 - 1530.0 * tmpr);
					ctmp2 = 0;
				}
				else if (tmpx2 < 0.5)
				{
					ctmp1 = 255.0;
					ctmp2 = (int)(1530.0 * tmpr);
					ctmp0 = 0;
				}
				else if (tmpx2 < 2.0 / 3.0)
				{
					ctmp2 = 255.0;
					ctmp1 = (int)(255.0 - 1530.0 * tmpr);
					ctmp0 = 0;
				}
				else if (tmpx2 < 5.0 / 6.0)
				{
					ctmp2 = 255.0;
					ctmp0 = (int)(1530.0 * tmpr);
					ctmp1 = 0;
				}
				else
				{
					ctmp0 = 255.0;
					ctmp2 = (int)(255.0 - 1530.0 * tmpr);
					ctmp1 = 0;
				}
			}
		}
		else
		{
			if (vecn1 < 0)
			{
				tmpx2 = (1.0 / 4.0) + (1.0 / 8.0) + coll0 / (8.0 * roomsize);
				tmpr = fmod(tmpx2, 1.0 / 6.0);

				if (tmpx2 < 1.0 / 6.0)
				{
					ctmp0 = 255.0;
					ctmp1 = (int)(1530.0 * tmpr);
					ctmp2 = 0;
				}
				else if (tmpx2 < 1.0 / 3.0)
				{
					ctmp1 = 255.0;
					ctmp0 = (int)(255.0 - 1530.0 * tmpr);
					ctmp2 = 0;
				}
				else if (tmpx2 < 0.5)
				{
					ctmp1 = 255.0;
					ctmp2 = (int)(1530.0 * tmpr);
					ctmp0 = 0;
				}
				else if (tmpx2 < 2.0 / 3.0)
				{
					ctmp2 = 255.0;
					ctmp1 = (int)(255.0 - 1530.0 * tmpr);
					ctmp0 = 0;
				}
				else if (tmpx2 < 5.0 / 6.0)
				{
					ctmp2 = 255.0;
					ctmp0 = (int)(1530.0 * tmpr);
					ctmp1 = 0;
				}
				else
				{
					ctmp0 = 255.0;
					ctmp2 = (int)(255.0 - 1530.0 * tmpr);
					ctmp1 = 0;
				}
			}
			else
			{
				tmpx2 = (3.0 / 4.0) + (1.0 / 8.0) - coll0 / (8.0 * roomsize);

				tmpr = fmod(tmpx2, 1.0 / 6.0);

				if (tmpx2 < 1.0 / 6.0)
				{
					ctmp0 = 255.0;
					ctmp1 = (int)(1530.0 * tmpr);
					ctmp2 = 0;
				}
				else if (tmpx2 < 1.0 / 3.0)
				{
					ctmp1 = 255.0;
					ctmp0 = (int)(255.0 - 1530.0 * tmpr);
					ctmp2 = 0;
				}
				else if (tmpx2 < 0.5)
				{
					ctmp1 = 255.0;
					ctmp2 = (int)(1530.0 * tmpr);
					ctmp0 = 0;
				}
				else if (tmpx2 < 2.0 / 3.0)
				{
					ctmp2 = 255.0;
					ctmp1 = (int)(255.0 - 1530.0 * tmpr);
					ctmp0 = 0;
				}
				else if (tmpx2 < 5.0 / 6.0)
				{
					ctmp2 = 255.0;
					ctmp0 = (int)(1530.0 * tmpr);
					ctmp1 = 0;
				}
				else
				{
					ctmp0 = 255.0;
					ctmp2 = (int)(255.0 - 1530.0 * tmpr);
					ctmp1 = 0;
				}
			}
		}

		if (checker == 0) {
			buffer[4 * tmp] = 0;
			buffer[4 * tmp + 1] = 0;
			buffer[4 * tmp + 2] = 0;

		}
		else
		{
			buffer[4 * tmp] = ctmp0;
			buffer[4 * tmp + 1] = ctmp1;
			buffer[4 * tmp + 2] = ctmp2;
		}
	}
}

void cudaInit()
{
	double dist = 2.0;
	double sqsz = 0.01 / 4;
	int tmpx, tmpy;
	double* vecltmp = new double[1920 * 1080];

	double vec0, vec1, vec2;
	double addy0, addy1, addy2;
	double addz0, addz1, addz2;
	double vecn0, vecn1, vecn2;
	double x00 = 1, x01 = 0, x02 = 0;
	double x10 = 0, x11 = 1, x12 = 0;
	double x20 = 0, x21 = 0, x22 = 1;
	double multy = (1 - 1920) * sqsz / 2;
	double multz = (1080 - 1) * sqsz / 2;

	cudaSetDevice(0);
	cudaMalloc((void**)&buffer, 4 * 1920 * 1080 * sizeof(uint8_t));
	cudaMalloc((void**)&vecl, 1920 * 1080 * sizeof(double));
	cudaMalloc((void**)&stars, starsize * starsize * sizeof(uint8_t));
	
	vec0 = dist * x00 + multy * x10 + multz * x20;
	vec1 = dist * x01 + multy * x11 + multz * x21;
	vec2 = dist * x02 + multy * x12 + multz * x22;
	
	addy0 = sqsz * x10;
	addy1 = sqsz * x11;
	addy2 = sqsz * x12;

	addz0 = -sqsz * x20;
	addz1 = -sqsz * x21;
	addz2 = -sqsz * x22;
	
	for (int i = 0; i < 1920 * 1080; i++)
	{
		tmpx = i % 1920;
		tmpy = (i - tmpx) / 1920;

		vecn0 = vec0 + tmpx * addy0 + tmpy * addz0;
		vecn1 = vec1 + tmpx * addy1 + tmpy * addz1;
		vecn2 = vec2 + tmpx * addy2 + tmpy * addz2;

		vecltmp[i] = sqrt(vecn0*vecn0+vecn1*vecn1+vecn2*vecn2);
	}

	cudaMemcpy(vecl, vecltmp, 1920 * 1080 * sizeof(double), cudaMemcpyHostToDevice);

	bufferinit << <(int)(1920 * 1080 / 600), 600 >> > (buffer);
	cudaDeviceSynchronize();

	setstars << <starsize * starsize / 500, 500 >> > (stars);
	cudaDeviceSynchronize();
}

void cudaExit()
{
	cudaFree(buffer);
	cudaFree(vecl);
	cudaDeviceReset();
}

void cudathingy(uint8_t* pixels, double pos0, double pos1, double pos2, double vec0, double vec1, double vec2, double addy0, double addy1, double addy2, double addz0, double addz1, double addz2, bool inside, double alpha, double beta, double bigr, double r, bool other)
{
	addKernel <<<(int)(1920 * 1080 / 600), 600>>>(buffer, vecl, pos0,pos1,pos2,vec0,vec1,vec2,addy0,addy1,addy2,addz0,addz1,addz2,inside,alpha, beta,bigr,  r,other,stars);

	cudaDeviceSynchronize();
	cudaMemcpy(pixels, buffer, 4 * 1920 * 1080 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
}
