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
	template<typename Real, typename Matrix>
	__device__ HyperelasticityModel<Real, Matrix>* getElasticityModel(EnergyType type)
	{
		switch (type)
		{
		case Linear:
			return new StVKModel<Real, Matrix>();
		case Quadratic:
			return NULL;
		default:
			break;
		}
	}

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
	__global__ void HM_ComputeF(
		DeviceArray<Matrix> F,
		DeviceArray<Coord> eigens,
		DeviceArray<Matrix> RU,
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
		F[pId] = matL_i*V*D*U.transpose();

		polarDecomposition(F[pId], R, U, D, V);

		eigens[pId] = Coord(D(0, 0), D(1, 1), D(2, 2));
		RU[pId] = U;
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
		F[pId] = matL_i*V*D*U.transpose();


// 		if (pId == 0)
// 		{
// 			printf("matL_i: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
// 				matL_i(0, 0), matL_i(0, 1), matL_i(0, 2),
// 				matL_i(1, 0), matL_i(1, 1), matL_i(1, 2),
// 				matL_i(2, 0), matL_i(2, 1), matL_i(2, 2));
// 
// 			printf("matK_i: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
// 				matK_i(0, 0), matK_i(0, 1), matK_i(0, 2),
// 				matK_i(1, 0), matK_i(1, 1), matK_i(1, 2),
// 				matK_i(2, 0), matK_i(2, 1), matK_i(2, 2));
// 
// 			printf("F: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
// 				F[pId](0, 0), F[pId](0, 1), F[pId](0, 2),
// 				F[pId](1, 0), F[pId](1, 1), F[pId](1, 2),
// 				F[pId](2, 0), F[pId](2, 1), F[pId](2, 2));
// 
// 			printf("inverseF: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
// 				inverseF[pId](0, 0), inverseF[pId](0, 1), inverseF[pId](0, 2),
// 				inverseF[pId](1, 0), inverseF[pId](1, 1), inverseF[pId](1, 2),
// 				inverseF[pId](2, 0), inverseF[pId](2, 1), inverseF[pId](2, 2));
// 
// 			printf("inverseL: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
// 				inverseL[pId](0, 0), inverseL[pId](0, 1), inverseL[pId](0, 2),
// 				inverseL[pId](1, 0), inverseL[pId](1, 1), inverseL[pId](1, 2),
// 				inverseL[pId](2, 0), inverseL[pId](2, 1), inverseL[pId](2, 2));
// 
// 			printf("inverseK: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
// 				inverseK[pId](0, 0), inverseK[pId](0, 1), inverseK[pId](0, 2),
// 				inverseK[pId](1, 0), inverseK[pId](1, 1), inverseK[pId](1, 2),
// 				inverseK[pId](2, 0), inverseK[pId](2, 1), inverseK[pId](2, 2));
// 		}
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
		stressTensor[pId] = F_i * (2 * lambda * E);// F_i * (2 * mu * E + lambda * E.trace() * Matrix::identityMatrix());
		//stressTensor[pId] = F_i * (2 * mu * E);
		//stressTensor[pId] = 2 * mu * F_i;
		//stressTensor[pId] = F_i * ( lambda * E.trace() * Matrix::identityMatrix());
//		stressTensor[pId] = Matrix(0);

//  		if (F_i.determinant() < 0.0f)
//  		{
// 			printf("F_i: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
// 				F_i(0, 0), F_i(0, 1), F_i(0, 2),
// 				F_i(1, 0), F_i(1, 1), F_i(1, 2),
// 				F_i(2, 0), F_i(2, 1), F_i(2, 2));
//  		}
// 		if (pId == 0)
// 		{
// 			printf("E_i: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
// 				E(0, 0), E(0, 1), E(0, 2),
// 				E(1, 0), E(1, 1), E(1, 2),
// 				E(2, 0), E(2, 1), E(2, 2));
// 
// 			printf("stressTensor: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
// 				stressTensor[pId](0, 0), stressTensor[pId](0, 1), stressTensor[pId](0, 2),
// 				stressTensor[pId](1, 0), stressTensor[pId](1, 1), stressTensor[pId](1, 2),
// 				stressTensor[pId](2, 0), stressTensor[pId](2, 1), stressTensor[pId](2, 2));
// 		}
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
		Real mu,
		Real lambda,
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
		Matrix PK_i = stressTensor[pId] * invK[pId];

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
				Matrix PK_j = stressTensor[j] * invK[j];

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


			Matrix K_i = invK[pId];
			printf("invK: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
				K_i(0, 0), K_i(0, 1), K_i(0, 2),
				K_i(1, 0), K_i(1, 1), K_i(1, 2),
				K_i(2, 0), K_i(2, 1), K_i(2, 2));
		}

	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_JacobiStep(
		DeviceArray<Coord> y_next,
		DeviceArray<Coord> y_pre,
		DeviceArray<Coord> y_old,
		DeviceArray<Matrix> Rot,
		DeviceArray<Coord> eigen,
		NeighborList<NPair> restShapes,
		Real horizon,
		DeviceArray<Real> volume,
		Real dt,
		EnergyType type)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		SmoothKernel<Real> kernSmooth;
		HyperelasticityModel<Real, Matrix>* model = getElasticityModel<Real, Matrix>(type);

		Real lambda_i1 = eigen[pId][0];
		Real lambda_i2 = eigen[pId][1];
		Real lambda_i3 = eigen[pId][2];

		Matrix PK1_i = Rot[pId] * model->getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3)*Rot[pId].transpose();
		Matrix PK2_i = model->getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3);

		Real V_i = volume[pId];

		Real mass_i = V_i*model->density;

		Coord y_pre_i = y_pre[pId];
		Coord y_rest_i = restShapes.getElement(pId, 0).pos;

		Matrix mat_i(0);
		Coord source_i(0);

		int size_i = restShapes.getNeighborSize(pId);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord y_pre_j = y_pre[j];
			Real r = (np_j.pos - y_rest_i).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);

				Real V_j = volume[j];

				const Real scale = V_i*V_j;

				const Real sw_ij = dt*dt*scale*weight;

				Real lambda_j1 = eigen[j][0];
				Real lambda_j2 = eigen[j][1];
				Real lambda_j3 = eigen[j][2];

				Matrix PK1_ij = PK1_i + Rot[j]*model->getStressTensorPositive(lambda_j1, lambda_j2, lambda_j3)*Rot[j].transpose();
				Matrix PK2_ij = PK2_i + model->getStressTensorNegative(lambda_j1, lambda_j2, lambda_j3);

				Coord y_pre_ij = (y_pre_i - y_pre_j);
				source_i += sw_ij*PK1_ij*y_pre_j + sw_ij*PK2_ij*y_pre_ij;
				mat_i += sw_ij*PK1_ij;
			}
		}

		Coord y_old_i = y_old[pId];
		source_i += mass_i*y_old_i;

		mat_i += mass_i*Matrix::identityMatrix();
		y_next[pId] = mat_i.inverse()*source_i;

		delete model;
	}

	template<typename TDataType>
	bool HyperelasticityModule_test<TDataType>::initializeImpl()
	{
		m_F.resize(this->m_position.getElementCount());
		m_invF.resize(this->m_position.getElementCount());

		m_eigenValues.resize(this->m_position.getElementCount());
		m_Rot.resize(this->m_position.getElementCount());
		m_volume.resize(this->m_position.getElementCount());

		m_energy.resize(this->m_position.getElementCount());
		m_gradient.resize(this->m_position.getElementCount());

		y_pre.resize(this->m_position.getElementCount());
		y_next.resize(this->m_position.getElementCount());

		initializeVolume();

		m_reduce = Reduction<Real>::Create(this->m_position.getElementCount());

		return ElasticityModule::initializeImpl();
	}

	template <typename Real>
	__global__ void HM_InitVolume(
		DeviceArray<Real> volume
	) 
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= volume.size()) return;

		volume[pId] = Real(1);
	}
	

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::initializeVolume()
	{
		int numOfParticles = this->m_position.getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		HM_InitVolume << <pDims, BLOCK_SIZE >> > (m_volume);
	}


	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticity()
	{
		solveElasticityImplicit();
	}

	int ind_num = 0;

	template <typename Coord, typename Matrix>
	__global__ void HM_RotateInitPos(
		DeviceArray<Coord> position)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		Matrix rotM(0);
		float theta = 3.1415926 / 6.0f;
		rotM(0, 0) = cos(theta);
		rotM(0, 1) = -sin(theta);
		rotM(1, 0) = sin(theta);
		rotM(1, 1) = cos(theta);
// 		rotM(0, 0) = sin(theta);
// 		rotM(0, 1) = cos(theta);
// 		rotM(1, 0) = cos(theta);
// 		rotM(1, 1) = -sin(theta);
		rotM(2, 2) = 1.0f;
		Coord origin = position[0];
		position[pId] = origin + rotM*(position[pId] - origin);
	}

	template <typename Coord>
	__global__ void HM_ComputeGradient(
		DeviceArray<Coord> grad,
		DeviceArray<Coord> y_pre,
		DeviceArray<Coord> y_next)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		grad[pId] = y_next[pId] - y_pre[pId];

//		printf("Thread ID %d: %f, %f, %f \n", pId, grad[pId][0], grad[pId][1], grad[pId][2]);
	}

	template <typename Real, typename Coord>
	__global__ void HM_ComputeCurrentPosition(
		DeviceArray<Coord> grad,
		DeviceArray<Coord> y_pre,
		DeviceArray<Coord> y_next,
		Real alpha)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		y_next[pId] = y_pre[pId] + alpha*grad[pId];
	}

	template <typename Real, typename Coord, typename NPair>
	__global__ void HM_Compute1DEnergy(
		DeviceArray<Real> energy,
		DeviceArray<Coord> pos_pre,
		DeviceArray<Coord> pos_current,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real mu,
		Real lambda)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= energy.size()) return;

		mu = 48000;
		lambda = 12000;

		SmoothKernel<Real> kernSmooth;

		Coord pos_pre_i = pos_pre[pId];
		Coord pos_current_i = pos_current[pId];

		Real totalEnergy = 0.0f;// (pos_current_i - pos_pre_i).normSquared();

		int size_i = restShapes.getNeighborSize(pId);

		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;

		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord pos_current_j = pos_current[j];
			Real r = (np_j.pos - rest_pos_i).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				Real stretch = (pos_current_j - pos_current_i).normSquared() / (r*r) - 1.0;

				totalEnergy += mu*weight*stretch*stretch;
			}
		}

		energy[pId] = totalEnergy;
	}

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticityImplicit()
	{
		int numOfParticles = this->m_position.getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		this->m_weights.reset();

		Log::sendMessage(Log::User, "\n \n \n \n *************solver start!!!***************");

		if (ind_num == 0)
		{
			HM_RotateInitPos <Coord, Matrix> << <pDims, BLOCK_SIZE >> > (this->m_position.getValue());
			ind_num++;
		}

		/**************************** Jacobi method ************************************************/
		// initialize y_now, y_next_iter
		Function1Pt::copy(y_pre, this->m_position.getValue());
		Function1Pt::copy(y_next, this->m_position.getValue());
		Function1Pt::copy(m_position_old, this->m_position.getValue());

		// do Jacobi method Loop
		bool convergeFlag = false; // converge or not
		int iterCount = 0;

		while (iterCount < 10) {

			HM_ComputeF << <pDims, BLOCK_SIZE >> > (
				m_F,
				m_eigenValues,
				m_Rot,
				y_pre,
				this->m_restShape.getValue(),
				this->m_horizon.getValue());
			cuSynchronize();

			HM_JacobiStep << <pDims, BLOCK_SIZE >> > (
				y_next,
				y_pre,
				m_position_old,
				m_Rot,
				m_eigenValues,
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				m_volume, this->getParent()->getDt(),
				m_energyType);

			HM_ComputeGradient << <pDims, BLOCK_SIZE >> > (
				m_gradient,
				y_pre,
				y_next);
			cuSynchronize();

			//stepsize adjustment
			{
				Real totalE_before;
				Real totalE_current;
				getEnergy(totalE_before, y_pre);
				getEnergy(totalE_current, y_next);

				printf("Previous: %f Next: %f \n", totalE_before, totalE_current);

				Real alpha = 1.0f;
				int step = 0;

				while (totalE_current > totalE_before + 100.0)
				{
					step++;
					alpha *= 0.5;
					
					printf("Previous: %f Next: %f \n", totalE_before, totalE_current);
					printf("Iteration %d Step %d alpha: %f \n", iterCount, step, alpha);

					HM_ComputeCurrentPosition << <pDims, BLOCK_SIZE >> > (
						m_gradient,
						y_pre,
						y_next,
						alpha);

					getEnergy(totalE_current, y_next);
				}

				Function1Pt::copy(y_pre, y_next);
			}

			iterCount++;
		}

		test_HM_UpdatePosition << <pDims, BLOCK_SIZE >> > (
			this->m_position.getValue(),
			this->m_velocity.getValue(),
			y_next,
			m_position_old,
			this->getParent()->getDt());
		cuSynchronize();
	}


	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticityGradientDescent()
	{

	}


	template <typename Real, typename Coord, typename Matrix>
	__global__ void HM_ComputeEnergy(
		DeviceArray<Real> energy,
		DeviceArray<Coord> eigens,
		EnergyType type)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= energy.size()) return;

		HyperelasticityModel<Real, Matrix>* model = getElasticityModel<Real, Matrix>(type);

		Coord eigen_i = eigens[pId];

		energy[pId] = model->getEnergy(eigen_i[0], eigen_i[1], eigen_i[2]);

		delete model;
	}

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::getEnergy(Real& totalEnergy, DeviceArray<Coord>& position)
	{
		int numOfParticles = this->m_position.getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		HM_ComputeF << <pDims, BLOCK_SIZE >> > (
			m_F,
			m_eigenValues,
			m_Rot,
			position,
			this->m_restShape.getValue(),
			this->m_horizon.getValue());

		HM_ComputeEnergy <Real, Coord, Matrix> << <pDims, BLOCK_SIZE >> > (
			m_energy,
			m_eigenValues,
			m_energyType);
		cuSynchronize();

		totalEnergy = m_reduce->accumulate(m_energy.getDataPtr(), m_energy.size());
	}


#ifdef PRECISION_FLOAT
	template class HyperelasticityModule_test<DataType3f>;
#else
	template class HyperelasticityModule_test<DataType3d>;
#endif
}