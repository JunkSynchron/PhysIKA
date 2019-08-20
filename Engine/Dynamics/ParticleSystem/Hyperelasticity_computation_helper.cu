#include "HyperelasticityModule_test.h"
#include "Core/Utility.h"
#include "Core/Algorithm/MatrixFunc.h"

#include "Kernel.h"
#include <math.h>



namespace Physika 
{
	template <typename NPair>
	__global__ void findNieghborNums(
		NeighborList<NPair>& restShapes,
		DeviceArray<int>& neighborNums)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= neighborNums.size()) return;

		// size_i include this particle itself
		int size_i = restShapes.getNeighborSize(pId);
		neighborNums[pId] = size_i;
	}


	//-test: to find all the deformation gradient matrices
	// these deformation gradients are mat3x3
	template <typename Real, typename Coord, typename Matrix, typename NPair>
	GPU_FUNC void getDeformationGradient(
		int curParticleID,
		DeviceArray<Coord>& position,
		NeighborList<NPair>& restShapes,
		Real horizon,
		Matrix& resultMatrix)
	{
		
		resultMatrix = Matrix(0.0f);

		CorrectedKernel<Real> g_weightKernel;

		NPair np_i = restShapes.getElement(curParticleID, 0);
		Coord rest_i = np_i.pos;
		int size_i = restShapes.getNeighborSize(curParticleID);

		Real total_weight = Real(0);
		Matrix deform_i = Matrix(0.0f);
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(curParticleID, ne);
			Coord rest_j = np_j.pos;
			int j = np_j.index;

			Real r = (rest_j - rest_i).norm();

			if (r > EPSILON)
			{
				Real weight = g_weightKernel.Weight(r, horizon);

				Coord p = (position[j] - position[curParticleID]) / horizon;
				Coord q = (rest_j - rest_i) / horizon * weight;

				deform_i(0, 0) += p[0] * q[0]; deform_i(0, 1) += p[0] * q[1]; deform_i(0, 2) += p[0] * q[2];
				deform_i(1, 0) += p[1] * q[0]; deform_i(1, 1) += p[1] * q[1]; deform_i(1, 2) += p[1] * q[2];
				deform_i(2, 0) += p[2] * q[0]; deform_i(2, 1) += p[2] * q[1]; deform_i(2, 2) += p[2] * q[2];
				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			deform_i *= (1.0f / total_weight);
		}
		else
		{
			total_weight = 1.0f;
		}

		resultMatrix = deform_i;

	}




	// -test: singular value decomposition
	// matrix A = U * S * V^T
	// matrix A are 3x3
	template <typename Matrix>
	GPU_FUNC void getSVDmatrix(
		const Matrix& A,
		Matrix& U,
		Matrix& S,
		Matrix& V) 
	{
		typedef typename Matrix::VarType Real;
		typedef typename Vector<Real, 3> Coord;


		int rowA = A.rows();
		int colA = A.cols();

		S = Matrix(0.0);
		U = A;
		V = Matrix::identityMatrix();

		// set the tolerance
		Real TOL = 1e-8;
		// current tolerance
		Real converge = TOL + 1;


		// Jacobi Rotation Loop
		// reference to http://www.math.pitt.edu/~sussmanm/2071Spring08/lab09/index.html
		while (converge > TOL) {
			converge = 0.0;
			for (int j = 1; j <= colA-1; ++j) {
				for (int i = 0; i <= j - 1; ++i) {
					// compute [alpha gamma; gamma beta]=(i,j) submatrix of U^T * U
					Real coeAlpha = Real(0);
					for (int k = 0; k <= colA - 1; ++k) {
						coeAlpha += U(k, i)*U(k, i);
					}
					Real coeBeta = Real(0);
					for (int k = 0; k <= colA - 1; ++k) {
						coeBeta += U(k, j)*U(k, j);
					}
					Real coeGamma = Real(0);
					for (int k = 0; k <= colA - 1; ++k) {
						coeGamma += U(k, i)*U(k, j);
					}

					// find current tolerance
					converge = max(converge, abs(coeGamma)/sqrt(coeAlpha*coeBeta));

					// compute Jacobi Rotation
					// take care Gamma may be zero
					Real coeZeta, coeTan;
					if (converge > TOL) {
						coeZeta = (coeBeta - coeAlpha) / (2 * coeGamma);
						int signOfZeta = (Real(0) < coeZeta) - (coeZeta < Real(0));
						coeTan = signOfZeta / (abs(coeZeta) + sqrt(1.0 + coeZeta * coeZeta));
					}
					else {
						coeTan = 0.0;
					}
					
					Real coeCos = Real(1.0) / (sqrt(1.0 + coeTan * coeTan));
					Real coeSin = coeCos * coeTan;

					// update columns i and j of U
					for (int k = 0; k <= colA - 1; ++k) {
						Real tmp = U(k, i);
						U(k, i) = coeCos * coeTan - coeSin * U(k, j);
						U(k, j) = coeSin * coeTan + coeCos * U(k, j);
					}
					//update columns of V
					for (int k = 0; k <= colA - 1; ++k) {
						Real tmp = V(k, i);
						V(k, i) = coeCos * coeTan - coeSin * V(k, j);
						V(k, j) = coeSin * coeTan + coeCos * V(k, j);
					}

				}
			}
		}

		// find singular values and normalize U
		Coord singValues = Coord(1.0);
		for (int j = 0; j <= colA - 1; ++j) {
			singValues[j] = U.col(j).norm();
			for (int i = 0; i <= rowA - 1; ++i) {
				U(i, j) = U(i, j) / singValues[j];
			}
		}

		// get matrix S
		for (int i = 0; i < rowA; ++i) {
			S(i, i) = singValues[i];
		}
	}

	template <typename Matrix>
	GPU_FUNC void getGInverseMatrix_SVD(
		Matrix& U,
		Matrix& S,
		Matrix& V,
		Matrix& resultMat)
	{
		typedef typename Matrix::VarType Real;
		typedef typename Vector<Real, 3> Coord;

		int colS = S.cols();

		// inverse matrix S
		for (int j = 0; j <= colS - 1; ++j) {
			if (S(j, j) > EPSILON) { S(j, j) = 1.0 / S(j, j); }
		}

		//transpose mat U
		U = U.transpose();

		// A = U*S*V^T;  so that A^-1 = V * S^-1 * U^T
		resultMat = V * S * U;
	}



	template <typename Matrix>
	GPU_FUNC void GInverseMat(
		Matrix& A,
		Matrix& resultMat) 
	{
		Matrix matU = Matrix(0.0), matV = Matrix(0.0), matS = Matrix(0.0);
		getSVDmatrix(A, matU, matS, matV);
		getGInverseMatrix_SVD(matU, matS, matV, resultMat);
	}


	//-test: to find generalized inverse of all deformation gradient matrices
	// these deformation gradients are mat3x3, may be singular
	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void get_GInverseOfF_PiolaKirchhoff(
		DeviceArray<Coord>& position,
		NeighborList<NPair>& restShapes,
		Real horizon,
		Real distance,
		Real mu,
		Real lambda,
		DeviceArray<Matrix>& resultGInverseMatrices,
		DeviceArray<Matrix>& firstPiolaKirchhoffMatrices)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		Real total_weight = Real(0);
		Matrix deform_i = Matrix(0.0f);
		getDeformationGradient(
			pId,
			position,
			restShapes,
			horizon,
			deform_i);
		
		// deformation gradients mat SVD
		Matrix matU = Matrix(0.0), matV = Matrix(0.0), matS = Matrix(0.0);
		getSVDmatrix(deform_i, matU, matS, matV);

		// get g-inverse of deformaition gradients F
		Matrix matInverseF = Matrix(0.0);
		getGInverseMatrix_SVD(matU, matS, matV, matInverseF);
		resultGInverseMatrices[pId] = matInverseF;

		// find strain tensor E = 1/2(F^T * F - I)
		Matrix strainMat = 0.5*(deform_i.transpose() * deform_i - Matrix::identityMatrix());
		// find first Piola-Kirchhoff matix
		firstPiolaKirchhoffMatrices[pId] = deform_i * (2 * mu*strainMat + lambda * strainMat.trace() * Matrix::identityMatrix());
	}


	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void getJacobiMethod_D_R_b_constants(
		DeviceArray<Coord>& position,
		NeighborList<NPair>& restShapes,
		DeviceArray<Coord>& velocity,

		Real horizon,
		Real mass,
		Real volume,
		Real dt,

		DeviceArray<Matrix>& deformGradGInverseMats,
		DeviceArray<Matrix>& PiolaKirchhoffMats,

		DeviceArray< DeviceArray<Matrix> >& arrayR,
		DeviceArray<Matrix>& arrayDiagInverse,
		DeviceArray<Coord>& array_b)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		int curParticleID = pId;

		CorrectedKernel<Real> g_weightKernel;

		NPair np_i = restShapes.getElement(curParticleID, 0);
		Coord rest_i = np_i.pos;
		// size_i include this particle itself
		int size_i = restShapes.getNeighborSize(curParticleID);

		Real total_weight = Real(0);

		// compute mat R
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(curParticleID, ne);
			Coord rest_j = np_j.pos;

			int j = np_j.index;

			Real r = (rest_j - rest_i).norm();
			Real weight;
			if (r > EPSILON)
			{
				// compute weights: w_ij w_ji
				weight = g_weightKernel.Weight(r, horizon);

				total_weight += weight;
			}
			else {
				weight = 0.0;
			}

			arrayR[curParticleID][ne] = (-1.0) * dt * dt* volume* volume* (
				weight * PiolaKirchhoffMats[curParticleID] * deformGradGInverseMats[curParticleID]
				+ weight * PiolaKirchhoffMats[j] * deformGradGInverseMats[j]);
		}

		if (total_weight > EPSILON)
		{
			for (int ne = 1; ne < size_i; ne++) {
				arrayR[curParticleID][ne] *= 1.0f / (total_weight);
			}
		}
		else
		{
			total_weight = 1.0f;
		}

		// compute mat D
		Matrix matDiag = mass * Matrix::identityMatrix();
		for (int ne = 1; ne < size_i; ne++) {
			matDiag += (-1.0)* arrayR[curParticleID][ne];
		}
		Matrix matDiagInverse = Matrix(0.0);
		GInverseMat(matDiag, matDiagInverse);
		arrayDiagInverse[curParticleID] = matDiagInverse;

		array_b[curParticleID] = mass * (position[curParticleID] + dt * velocity[curParticleID]);
	}

	
	// one iteration of Jacobi method 
	template <typename Coord, typename Matrix, typename NPair>
	__global__ void JacobiStep(
		DeviceArray<DeviceArray<Matrix>>& arrayR,
		DeviceArray<Matrix>& arrayDiagInverse,
		DeviceArray<Coord>& array_b,
		
		NeighborList<NPair>& restShapes,

		DeviceArray<Coord>& y_pre,
		DeviceArray<Coord>& y_next) 
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_pre.size()) return;

		NPair np_i = restShapes.getElement(pId, 0);

		// size_i include this particle itself
		int size_i = restShapes.getNeighborSize(pId);

		Coord sigma = Coord(0.0);
		for (int ne = 1; ne < size_i; ++ne) {
			int index_j = restShapes.getElement(pId, ne).index;
			Matrix remainder = arrayR[pId][ne];
			sigma += remainder * y_pre[index_j];
		}
		y_next[pId] = arrayDiagInverse[pId] * (array_b[pId] - sigma);
	}

	// copy src to dst
	template <typename Coord>
	__global__ void arrayCopy(
		DeviceArray<Coord>& src,
		DeviceArray<Coord>& dst)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= src.size()) return;

		dst[pId] = src[pId];
	}

}
