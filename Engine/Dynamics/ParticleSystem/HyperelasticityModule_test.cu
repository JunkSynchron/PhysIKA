#include "HyperelasticityModule_test.h"
#include "Core/Utility.h"
#include "Framework/Framework/Node.h"
#include "Core/Algorithm/MatrixFunc.h"
#include "Kernel.h"

#include "Framework/Framework/Log.h"
#include "Core/Utility/Function1Pt.h"

#include "Hyperelasticity_computation_helper.cu"

namespace Physika
{
	template<typename TDataType>
	HyperelasticityModule_test<TDataType>::HyperelasticityModule_test()
		: ElasticityModule<TDataType>()
		, m_energyType(Linear)
	{
	}

	template <typename Coord>
	__global__ void test_HM_UpdatePosition(
		DeviceArray<Coord> position,
		DeviceArray<Coord> y_next
		)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		position[pId] = y_next[pId];
	}

	template <typename Real, typename Coord>
	__global__ void HM_ComputeSourceTerm(
		DeviceArray<Coord> source,
		DeviceArray<Coord> position,
		DeviceArray<Coord> velocity,
		Real mass,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		source[pId] = mass * (position[pId] + dt*velocity[pId]);
	}


	//-test: to find generalized inverse of all deformation gradient matrices
	// these deformation gradients are mat3x3, may be singular
	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_ComputeFandInverse(
		DeviceArray<Matrix> F,
		DeviceArray<Matrix> inverseF,
		DeviceArray<Coord> position,
		NeighborList<NPair> restShapes,
		Real horizon)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		SmoothKernel<Real> kernSmooth;

		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		int size_i = restShapes.getNeighborSize(pId);

		Real total_weight = Real(0);
		Matrix matF_i(0);
		Matrix matK_i(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);

				Coord p = (position[j] - position[pId]) / horizon;
				Coord q = (rest_pos_j - rest_pos_i) / horizon;

				matF_i(0, 0) += p[0] * q[0] * weight; matF_i(0, 1) += p[0] * q[1] * weight; matF_i(0, 2) += p[0] * q[2] * weight;
				matF_i(1, 0) += p[1] * q[0] * weight; matF_i(1, 1) += p[1] * q[1] * weight; matF_i(1, 2) += p[1] * q[2] * weight;
				matF_i(2, 0) += p[2] * q[0] * weight; matF_i(2, 1) += p[2] * q[1] * weight; matF_i(2, 2) += p[2] * q[2] * weight;

				matK_i(0, 0) += q[0] * q[0] * weight; matK_i(0, 1) += q[0] * q[1] * weight; matK_i(0, 2) += q[0] * q[2] * weight;
				matK_i(1, 0) += q[1] * q[0] * weight; matK_i(1, 1) += q[1] * q[1] * weight; matK_i(1, 2) += q[1] * q[2] * weight;
				matK_i(2, 0) += q[2] * q[0] * weight; matK_i(2, 1) += q[2] * q[1] * weight; matK_i(2, 2) += q[2] * q[2] * weight;

				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			matF_i *= (1.0f / total_weight);
			matK_i *= (1.0f / total_weight);
		}

		Matrix R, U, D, V;
		polarDecomposition(matK_i, R, U, D, V);

		Real threshold = 0.0001f*horizon;
		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;

		F[pId] = matF_i*V*D*U.transpose();

		polarDecomposition(matF_i, R, U, D, V);
		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;
		inverseF[pId] = matK_i*V*D*U.transpose();
	}

	template <typename Real, typename Matrix>
	__global__ void HM_ComputeFirstPiolaKirchhoff(
		DeviceArray<Matrix> stressTensor,
		DeviceArray<Matrix> F,
		DeviceArray<Matrix> inverseF,
		Real mu,
		Real lambda)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= inverseF.size()) return;

		Matrix F_i = F[pId];

		// find strain tensor E = 1/2(F^T * F - I)
		Matrix E = 0.5*(F_i.transpose() * F_i - Matrix::identityMatrix());
		// find first Piola-Kirchhoff matix; StVK material
		stressTensor[pId] = F_i * (2 * mu * E + lambda * E.trace() * Matrix::identityMatrix());
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_JacobiStep(
		DeviceArray<Coord> y_new,
		DeviceArray<Coord> y_old,
		DeviceArray<Coord> source,
		DeviceArray<Matrix> stressTensor,
		DeviceArray<Matrix> invF,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real mass,
		Real volume,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_old.size()) return;

		SmoothKernel<Real> kernSmooth;

		const Real scale = dt*dt*volume*volume;

		int size_i = restShapes.getNeighborSize(pId);

		Coord y_i = y_old[pId];
		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		Matrix PF_i = stressTensor[pId]* invF[pId];

		Matrix mat_i(0);
		Coord source_i = source[pId];
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord y_j = y_old[j];
			Matrix ST_j = stressTensor[j];
			Matrix invF_j = invF[j];
			Real r = (np_j.pos - rest_pos_i).norm();


			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);

				Matrix PF_j = stressTensor[j] * invF[j];

				Matrix PF_ij = scale * (PF_i + PF_j);

				mat_i += PF_ij;

				source_i += PF_ij * (y_j - y_i);
			}
		}

		mat_i += mass*Matrix::identityMatrix();

		y_new[pId] = mat_i.inverse() * source_i;
	}


	template<typename TDataType>
	bool HyperelasticityModule_test<TDataType>::initializeImpl()
	{
		m_F.resize(this->m_position.getElementCount());
		m_invF.resize(this->m_position.getElementCount());
		m_firstPiolaKirchhoffStress.resize(this->m_position.getElementCount());

		return ElasticityModule::initializeImpl();
	}



	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticity()
	{
		int numOfParticles = this->m_position.getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		this->m_displacement.reset();
		this->m_weights.reset();

		Log::sendMessage(Log::User, "solver start!!!");

		/****************************************************************************************************/
		//-test: compute the g-inverse deformation tensor & Piola-Kirchhoff tensor

		// all the g-inverse matices of deformation gradients F
		DeviceArray<Matrix> GInverseMatrices(numOfParticles);
		// all the first Piola-Kirchhoff tensors
		DeviceArray<Matrix> FirstPiolaKirchhoffMatrices(numOfParticles);


		// mass and volume are set 1.0, (need modified) 
		Real mass = 1.0;
		Real volume = 1.0;

		DeviceArray<Coord> array_b(numOfParticles);
		HM_ComputeSourceTerm << <pDims, BLOCK_SIZE >> > (
			array_b,
			this->m_position.getValue(),
			this->m_velocity.getValue(),
			mass,
			this->getParent()->getDt());


		/**************************** Jacobi method ************************************************/
		// compute constants of the linear equations

		// find out i-th particle's all neighbors and compute diagonal component D & remainder sparse matrix R

		// find size_i: number of neighbors of i-th particle


		DeviceArray<int>& arrayRIndex = this->m_restShape.getValue().getIndex();
		int arrayRSize = this->m_restShape.getValue().getElementsSize();

		// allocate memory for arrayR, use 1-dim array store 2-dim sparse matrix
		DeviceArray<Matrix> arrayR(arrayRSize);

		// we stored inverse of mat D
		DeviceArray<Matrix> arrayDiagInverse(numOfParticles);

		// release no use array
		GInverseMatrices.release();
		FirstPiolaKirchhoffMatrices.release();


		// initialize y_now, y_next_iter
		DeviceArray<Coord> y_pre(numOfParticles);
		DeviceArray<Coord> y_next(numOfParticles);

		Function1Pt::copy(y_pre, this->m_position.getValue());

		// do Jacobi method Loop
		bool convergeFlag = false; // converge or not
		int iterCount = 0;

		while (iterCount < 5) {

			HM_ComputeFandInverse << <pDims, BLOCK_SIZE >> > (
				m_F,
				m_invF,
				this->m_position.getValue(),
				this->m_restShape.getValue(),
				this->m_horizon.getValue());
			cuSynchronize();

			HM_ComputeFirstPiolaKirchhoff << <pDims, BLOCK_SIZE >> > (
				m_firstPiolaKirchhoffStress,
				m_F,
				m_invF,
				this->m_mu.getValue(),
				this->m_lambda.getValue());
			cuSynchronize();

			HM_JacobiStep << <pDims, BLOCK_SIZE >> > (
				y_next,
				y_pre,
				array_b,
				m_firstPiolaKirchhoffStress,
				m_invF,
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				mass, volume, this->getParent()->getDt());

			Function1Pt::copy(y_pre, y_next);

			iterCount++;
		}

		arrayR.release();
		array_b.release();
		arrayDiagInverse.release();

		test_HM_UpdatePosition << <pDims, BLOCK_SIZE >> > (
			this->m_position.getValue(),
			y_next
			);
		cuSynchronize();

		y_pre.release();
		y_next.release();
	}




#ifdef PRECISION_FLOAT
	template class HyperelasticityModule_test<DataType3f>;
#else
	template class HyperelasticityModule_test<DataType3d>;
#endif
}