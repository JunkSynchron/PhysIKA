#include "HyperelasticityModule_test.h"
#include "Core/Utility.h"
#include "Framework/Framework/Node.h"
#include "Core/Algorithm/MatrixFunc.h"
#include "Kernel.h"

#include "Hyperelasticity_computation_helper.cu"

namespace Physika
{
	template<typename TDataType>
	HyperelasticityModule_test<TDataType>::HyperelasticityModule_test()
		: ElasticityModule<TDataType>()
		, m_energyType(Linear)
	{
	}

	// previous method
	/************************************************************************************************
	template <typename Real, typename Coord, typename Matrix, typename NPair, typename Function>
	__global__ void HM_EnforceElasticity(
		DeviceArray<Coord> delta_position,
		DeviceArray<Real> weights,
		DeviceArray<Real> bulkCoefs,
		DeviceArray<Matrix> invK,
		DeviceArray<Coord> position,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real distance,
		Real mu,
		Real lambda,
		Function func)
	{

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		CorrectedKernel<Real> g_weightKernel;

		NPair np_i = restShapes.getElement(pId, 0);
		Coord rest_i = np_i.pos;
		int size_i = restShapes.getNeighborSize(pId);

		Coord cur_pos_i = position[pId];

		Coord accPos = Coord(0);
		Real accA = Real(0);
		Real bulk_i = bulkCoefs[pId];

		//compute the first invariant
		Real I1_i = Real(0);
		Real total_weight = Real(0);
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			Coord rest_pos_j = np_j.pos;
			int j = np_j.index;
			Real r = (rest_i - rest_pos_j).norm();

			if (r > 0.01*horizon)
			{
				Real weight = g_weightKernel.Weight(r, horizon);
				Coord p = (position[j] - cur_pos_i);
				Real ratio_ij = p.norm() / r;

				I1_i += weight * ratio_ij;

				total_weight += weight;
			}
		}

		I1_i = total_weight > EPSILON ? I1_i /= total_weight : Real(1);

		//compute the deformation tensor
		Matrix deform_i = Matrix(0.0f);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			Coord rest_j = np_j.pos;
			int j = np_j.index;

			Real r = (rest_j - rest_i).norm();

			if (r > EPSILON)
			{
				Real weight = g_weightKernel.Weight(r, horizon);

				Coord p = (position[j] - position[pId]) / horizon;
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
			deform_i = deform_i * invK[pId];
		}
		else
		{
			total_weight = 1.0f;
		}

		//Check whether the reference shape is inverted, if yes, simply set K^{-1} to be an identity matrix
		//Note other solutions are possible.
		if ((deform_i.determinant()) < -0.001f)
		{
			deform_i = Matrix::identityMatrix();
		}


		//solve the elasticity with projective peridynamics
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			Coord rest_j = np_j.pos;
			int j = np_j.index;
			Real r = (rest_j - rest_i).norm();

			Coord cur_pos_j = position[j];

			if (r > 0.01f*horizon)
			{
				Real weight = g_weightKernel.WeightRR(r, horizon);

				Coord rest_dir_ij = deform_i * (rest_i - rest_j);
				Coord cur_dir_ij = cur_pos_i - cur_pos_j;

				cur_dir_ij = cur_dir_ij.norm() > EPSILON ? cur_dir_ij.normalize() : Coord(0);
				rest_dir_ij = rest_dir_ij.norm() > EPSILON ? rest_dir_ij.normalize() : Coord(0, 0, 0);

				Real tau_ij = cur_dir_ij.norm() / r;

				Real mu_ij = mu * bulk_i* func(tau_ij) * g_weightKernel.WeightRR(r, horizon);
				Coord mu_pos_ij = position[j] + r * rest_dir_ij;
				Coord mu_pos_ji = position[pId] - r * rest_dir_ij;

				Real lambda_ij = lambda * bulk_i*func(I1_i)*g_weightKernel.WeightRR(r, horizon);
				Coord lambda_pos_ij = position[j] + r * cur_dir_ij;
				Coord lambda_pos_ji = position[pId] - r * cur_dir_ij;

				Coord delta_pos_ij = mu_ij * mu_pos_ij + lambda_ij * lambda_pos_ij;
				Real delta_weight_ij = mu_ij + lambda_ij;

				Coord delta_pos_ji = mu_ij * mu_pos_ji + lambda_ij * lambda_pos_ji;

				accA += delta_weight_ij;
				accPos += delta_pos_ij;


				atomicAdd(&weights[j], delta_weight_ij);
				atomicAdd(&delta_position[j][0], delta_pos_ji[0]);
				atomicAdd(&delta_position[j][1], delta_pos_ji[1]);
				atomicAdd(&delta_position[j][2], delta_pos_ji[2]);
			}
		}

		atomicAdd(&weights[pId], accA);
		atomicAdd(&delta_position[pId][0], accPos[0]);
		atomicAdd(&delta_position[pId][1], accPos[1]);
		atomicAdd(&delta_position[pId][2], accPos[2]);
	}

	template <typename Real, typename Coord>
	__global__ void HM_UpdatePosition(
		DeviceArray<Coord> position,
		DeviceArray<Coord> old_position,
		DeviceArray<Coord> delta_position,
		DeviceArray<Real> delta_weights)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		position[pId] = (old_position[pId] + delta_position[pId]) / (1.0 + delta_weights[pId]);
	}

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::previous_enforceElasticity()
	{
		int num = this->m_position.getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		this->m_displacement.reset();
		this->m_weights.reset();

		switch (m_energyType)
		{
		case Linear:
			HM_EnforceElasticity << <pDims, BLOCK_SIZE >> > (
				this->m_displacement,
				this->m_weights,
				this->m_bulkCoefs,
				this->m_invK,
				this->m_position.getValue(),
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				this->m_distance.getValue(),
				this->m_mu.getValue(),
				this->m_lambda.getValue(),
				ConstantFunc<Real>());
			cuSynchronize();
			break;

		case Quadratic:
			HM_EnforceElasticity << <pDims, BLOCK_SIZE >> > (
				this->m_displacement,
				this->m_weights,
				this->m_bulkCoefs,
				this->m_invK,
				this->m_position.getValue(),
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				this->m_distance.getValue(),
				this->m_mu.getValue(),
				this->m_lambda.getValue(),
				QuadraticFunc<Real>());
			cuSynchronize();
			break;

		default:
			break;
		}

		HM_UpdatePosition << <pDims, BLOCK_SIZE >> > (
			this->m_position.getValue(),
			this->m_position_old,
			this->m_displacement,
			this->m_weights);
		cuSynchronize();
	}
	***************************************************************************************/

	template <typename Real, typename Coord>
	__global__ void test_HM_UpdatePosition_Velocity(
		DeviceArray<Coord>& position,
		DeviceArray<Coord>& velocity,
		DeviceArray<Coord>& y_next,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		velocity[pId] = (y_next[pId] - position[pId]) / dt;
		position[pId] = y_next[pId];
	}



	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::enforceElasticity()
	{
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef TPair<TDataType> NPair;

		int num = this->m_position.getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		this->m_displacement.reset();
		this->m_weights.reset();

		int numParticles = num;


		/****************************************************************************************************/
		//-test: compute the g-inverse deformation tensor & Piola-Kirchhoff tensor

		// all the g-inverse matices of deformation gradients F
		DeviceArray<Matrix> GInverseMatrices(numParticles);
		// all the first Piola-Kirchhoff tensors
		DeviceArray<Matrix> FirstPiolaKirchhoffMatrices(numParticles);

		get_GInverseOfF_PiolaKirchhoff << <pDims, BLOCK_SIZE >> >(
			this->m_position.getValue(),
			this->m_restShape.getValue(),
			this->m_horizon.getValue(),
			this->m_distance.getValue(),
			this->m_mu.getValue(),
			this->m_lambda.getValue(),
			GInverseMatrices,
			FirstPiolaKirchhoffMatrices);
		cuSynchronize();

		// mass and volume are set 1.0, (need modified) 
		Real mass = 1.0;
		Real volume = 1.0;


		/**************************** Jacobi method ************************************************/
		// compute constants of the linear equations

		// find out i-th particle's all neighbors and compute diagonal component D & remainder sparse matrix R
		// arrayR[i][0] is no definition, i-th particle's neighbors are stored in arrayR[i][1:]
		DeviceArray<DeviceArray<Matrix>> arrayR(numParticles);
			//allocate mamery for arrayR

		// find size_i: number of neighbors of i-th particle
		DeviceArray<int> neighborNums(numParticles);
		findNieghborNums << <pDims, BLOCK_SIZE >> >(
			this->m_restShape.getValue(),
			neighborNums);
		cuSynchronize();

		// allocate memory for arrayR
		for (int i = 0; i < numParticles; ++i) {
			arrayR[i].resize(neighborNums[i]);
		}

		// we stored inverse of mat D
		DeviceArray<Matrix> arrayDiagInverse(numParticles);
		DeviceArray<Coord> array_b(numParticles);

		getJacobiMethod_D_R_b_constants << <pDims, BLOCK_SIZE >> >(
			this->m_position.getValue(),
			this->m_restShape.getValue(), 
			this->m_velocity.getValue(),

			this->m_horizon.getValue(),
			mass,
			volume,
			this->getParent()->getDt(),

			GInverseMatrices,
			FirstPiolaKirchhoffMatrices,
			arrayR,
			arrayDiagInverse,
			array_b); 
		cuSynchronize();
 
		// release no use array
		GInverseMatrices.release();
		FirstPiolaKirchhoffMatrices.release();


		// initialize y_now, y_next_iter
		DeviceArray<Coord> y_pre(numParticles);
		DeviceArray<Coord> y_next(numParticles);

		arrayCopy << <pDims, BLOCK_SIZE >> >(this->m_position.getValue(), y_pre);
		cuSynchronize();

		// do Jacobi method Loop
		bool convergeFlag = false; // converge or not
		int iterCount = 0;

		while (!convergeFlag) {
			JacobiStep << <pDims, BLOCK_SIZE >> > (
				arrayR,
				arrayDiagInverse,
				array_b,
				this->m_restShape.getValue(),
				y_pre,
				y_next);
			cuSynchronize();

			arrayCopy << <pDims, BLOCK_SIZE >> >(y_next, y_pre);
			cuSynchronize();

			iterCount++;
			if (iterCount > 15) { convergeFlag = true; }
		}

		test_HM_UpdatePosition_Velocity << <pDims, BLOCK_SIZE >> > (
			this->m_position.getValue(),
			this->m_velocity.getValue(),
			y_next,
			this->getParent()->getDt() );
		cuSynchronize();
	}



#ifdef PRECISION_FLOAT
	template class HyperelasticityModule_test<DataType3f>;
#else
	template class HyperelasticityModule_test<DataType3d>;
#endif
}