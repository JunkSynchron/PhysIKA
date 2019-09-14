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
		DeviceArray<Coord> velocity,
		DeviceArray<Coord> y_next,
		DeviceArray<Coord> position_old,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		position[pId] = y_next[pId];
		velocity[pId] += (position[pId] - position_old[pId]) / dt;
	}


	//-test: to find generalized inverse of all deformation gradient matrices
	// these deformation gradients are mat3x3, may be singular
	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_ComputeFandInverse(
		DeviceArray<Matrix> inverseK,
		DeviceArray<Matrix> inverseL,
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
		Matrix matL_i(0);
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

				matL_i(0, 0) += p[0] * q[0] * weight; matL_i(0, 1) += p[0] * q[1] * weight; matL_i(0, 2) += p[0] * q[2] * weight;
				matL_i(1, 0) += p[1] * q[0] * weight; matL_i(1, 1) += p[1] * q[1] * weight; matL_i(1, 2) += p[1] * q[2] * weight;
				matL_i(2, 0) += p[2] * q[0] * weight; matL_i(2, 1) += p[2] * q[1] * weight; matL_i(2, 2) += p[2] * q[2] * weight;

				matK_i(0, 0) += q[0] * q[0] * weight; matK_i(0, 1) += q[0] * q[1] * weight; matK_i(0, 2) += q[0] * q[2] * weight;
				matK_i(1, 0) += q[1] * q[0] * weight; matK_i(1, 1) += q[1] * q[1] * weight; matK_i(1, 2) += q[1] * q[2] * weight;
				matK_i(2, 0) += q[2] * q[0] * weight; matK_i(2, 1) += q[2] * q[1] * weight; matK_i(2, 2) += q[2] * q[2] * weight;

				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			matL_i *= (1.0f / total_weight);
			matK_i *= (1.0f / total_weight);
		}

		Matrix R, U, D, V;
		polarDecomposition(matK_i, R, U, D, V);

		Real threshold = 0.0001f*horizon;
		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;
		inverseK[pId] = V*D*U.transpose();
		F[pId] = matL_i*V*D*U.transpose();

		polarDecomposition(matL_i, R, U, D, V);
		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;
		inverseL[pId] = V*D*U.transpose();
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
		//stressTensor[pId] = F_i * ( lambda * E.trace() * Matrix::identityMatrix());


		if (pId == 0)
		{
			printf("trace %f \n", E.trace());
			printf("Mat: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
				F_i(0, 0), F_i(0, 1), F_i(0, 2),
				F_i(1, 0), F_i(1, 1), F_i(1, 2),
				F_i(2, 0), F_i(2, 1), F_i(2, 2));
		}
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_JacobiStepExplicit(
		DeviceArray<Coord> velocity,
		DeviceArray<Coord> y_new,
		DeviceArray<Coord> y_old,
		DeviceArray<Coord> source,
		DeviceArray<Matrix> stressTensor,
		DeviceArray<Matrix> invK,
		DeviceArray<Matrix> invL,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real mass,
		Real volume,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_old.size()) return;

		SmoothKernel<Real> kernSmooth;

		const Real scale = volume*volume;

		int size_i = restShapes.getNeighborSize(pId);

		Coord y_i = y_old[pId];
		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		//Matrix PK_i = stressTensor[pId] * invK[pId];
		Matrix PK_i = stressTensor[pId] * invL[pId];

		Matrix mat_i(0);
		Coord source_i = source[pId];
		Coord force_i(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord y_j = y_old[j];
			Real r = (np_j.pos - rest_pos_i).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);

				//Matrix PK_j = stressTensor[j] * invK[j];
				Matrix PK_j = stressTensor[j] * invL[j];

				Matrix PK_ij = scale * weight * (PK_i + PK_j);

				//force_i += PK_ij * (np_j.pos - rest_pos_i);
				force_i += PK_ij * (y_old[j] - y_old[pId]);
			}
		}


		velocity[pId] += force_i * dt / mass;

		if (pId == 0)
		{
			printf("force: %f %f %f \n", force_i[0], force_i[1], force_i[2]);

			printf("PK: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
				PK_i(0, 0), PK_i(0, 1), PK_i(0, 2),
				PK_i(1, 0), PK_i(1, 1), PK_i(1, 2),
				PK_i(2, 0), PK_i(2, 1), PK_i(2, 2));
		}

	}


	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_JacobiStep(
		DeviceArray<Coord> y_new,
		DeviceArray<Coord> y_old,
		DeviceArray<Coord> v_old,
		DeviceArray<Matrix> stressTensor,
		DeviceArray<Matrix> invL,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real mass,
		Real volume,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_old.size()) return;

		SmoothKernel<Real> kernSmooth;

		const Real scale = volume*volume;

		int size_i = restShapes.getNeighborSize(pId);

		Coord y_i = y_old[pId];
		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		//Matrix PK_i = stressTensor[pId] * invK[pId];
		Matrix PK_i = stressTensor[pId] * invL[pId];

		Matrix mat_i(0);
		Coord mv_i = v_old[pId];
		Coord totalSource_i = mass*mv_i;
		Coord dy_i(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord y_j = y_old[j];
			Real r = (np_j.pos - rest_pos_i).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);

				Matrix PK_j = stressTensor[j] * invL[j];

				Matrix PK_ij = dt*dt* scale * weight * (PK_i + PK_j);
				mat_i += PK_ij;

				//totalSource_i += PK_ij*(y_old[j]-y_old[pId]);
				totalSource_i += PK_ij*(y_old[j]);
			}
		}

		mat_i += mass*Matrix::identityMatrix();
		y_new[pId] = mat_i.inverse()*totalSource_i;
	}

	template<typename TDataType>
	bool HyperelasticityModule_test<TDataType>::initializeImpl()
	{
		m_position_old.resize(this->m_position.getElementCount());
		m_F.resize(this->m_position.getElementCount());
		m_invK.resize(this->m_position.getElementCount());
		m_invF.resize(this->m_position.getElementCount());
		m_invL.resize(this->m_position.getElementCount());
		m_firstPiolaKirchhoffStress.resize(this->m_position.getElementCount());

		return ElasticityModule::initializeImpl();
	}

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticity()
	{
		solveElasticityExplicit();
	}


	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticityExplicit()
	{
		int numOfParticles = this->m_position.getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		this->m_displacement.reset();
		this->m_weights.reset();

		Log::sendMessage(Log::User, "solver start!!!");

		/****************************************************************************************************/
		//-test: compute the g-inverse deformation tensor & Piola-Kirchhoff tensor


		// mass and volume are set 1.0, (need modified) 
		Real mass = 1.0;
		Real volume = 1.0;


		/**************************** Jacobi method ************************************************/
		// compute constants of the linear equations

		// find out i-th particle's all neighbors and compute diagonal component D & remainder sparse matrix R

		// find size_i: number of neighbors of i-th particle

		// initialize y_now, y_next_iter
		DeviceArray<Coord> y_pre(numOfParticles);
		DeviceArray<Coord> y_next(numOfParticles);

		Function1Pt::copy(y_pre, this->m_position.getValue());
		Function1Pt::copy(y_next, this->m_position.getValue());
		Function1Pt::copy(m_position_old, this->m_position.getValue());

		// do Jacobi method Loop
		bool convergeFlag = false; // converge or not
		int iterCount = 0;

		while (iterCount < 1) {

			HM_ComputeFandInverse << <pDims, BLOCK_SIZE >> > (
				m_invK,
				m_invL,
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

			// 			HM_JacobiStep << <pDims, BLOCK_SIZE >> > (
			// 				y_next,
			// 				y_pre,
			// 				m_position_old,
			// 				m_firstPiolaKirchhoffStress,
			// 				m_invL,
			// 				this->m_restShape.getValue(),
			// 				this->m_horizon.getValue(),
			// 				mass, volume, this->getParent()->getDt());

			HM_JacobiStepExplicit << <pDims, BLOCK_SIZE >> > (
				this->m_velocity.getValue(),
				y_next,
				y_pre,
				m_position_old,
				m_firstPiolaKirchhoffStress,
				m_invK,
				m_invL,
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				mass, volume, this->getParent()->getDt());

			Function1Pt::copy(y_pre, y_next);

			iterCount++;
		}

		y_pre.release();
		y_next.release();
	}


	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticityImplicit()
	{
		int numOfParticles = this->m_position.getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		this->m_displacement.reset();
		this->m_weights.reset();

		Log::sendMessage(Log::User, "solver start!!!");

		/****************************************************************************************************/
		//-test: compute the g-inverse deformation tensor & Piola-Kirchhoff tensor


		// mass and volume are set 1.0, (need modified) 
		Real mass = 1.0;
		Real volume = 1.0;


		/**************************** Jacobi method ************************************************/
		// compute constants of the linear equations

		// find out i-th particle's all neighbors and compute diagonal component D & remainder sparse matrix R

		// find size_i: number of neighbors of i-th particle

		// initialize y_now, y_next_iter
		DeviceArray<Coord> y_pre(numOfParticles);
		DeviceArray<Coord> y_next(numOfParticles);

		Function1Pt::copy(y_pre, this->m_position.getValue());
		Function1Pt::copy(y_next, this->m_position.getValue());
		Function1Pt::copy(m_position_old, this->m_position.getValue());

		// do Jacobi method Loop
		bool convergeFlag = false; // converge or not
		int iterCount = 0;

		while (iterCount < 10) {

			HM_ComputeFandInverse << <pDims, BLOCK_SIZE >> > (
				m_invK,
				m_invL,
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
				m_position_old,
				m_firstPiolaKirchhoffStress,
				m_invL,
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				mass, volume, this->getParent()->getDt());

			Function1Pt::copy(y_pre, y_next);

			iterCount++;
		}

		test_HM_UpdatePosition << <pDims, BLOCK_SIZE >> > (
			this->m_position.getValue(),
			this->m_velocity.getValue(),
			y_next,
			m_position_old,
			this->getParent()->getDt());
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