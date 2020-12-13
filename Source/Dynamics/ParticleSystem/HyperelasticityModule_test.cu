#include "HyperelasticityModule_test.h"
#include "Core/Utility.h"
#include "Core/Utility/CudaRand.h"
#include "Framework/Framework/Node.h"
#include "Core/Algorithm/MatrixFunc.h"
#include "Kernel.h"

#include "Framework/Framework/Log.h"
#include "Core/Utility/Function1Pt.h"

#include "Hyperelasticity_computation_helper.cu"

namespace PhysIKA
{
#define FIXEDNUM 4

	template<typename Real, typename Matrix>
	__device__ HyperelasticityModel<Real, Matrix>* getElasticityModel(EnergyType type)
	{
		switch (type)
		{
		case StVK:
			return new StVKModel<Real, Matrix>();
		case NeoHooekean:
			return new NeoHookeanModel<Real, Matrix>();
		case Polynomial:
			return new PolynomialModel<Real, Matrix, 1>();
		case Xuetal:
			return new XuModel<Real, Matrix>();
		default:
			break;
		}
	}

	template<typename TDataType>
	HyperelasticityModule_test<TDataType>::HyperelasticityModule_test()
		: ElasticityModule<TDataType>()
		, m_energyType(Xuetal)
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
		DeviceArray<Matrix> invK,
		DeviceArray<Matrix> matU,
		DeviceArray<Matrix> matV,
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
		invK[pId] = V*D*U.transpose();
		Matrix F_i = matL_i*V*D*U.transpose();

		polarDecomposition(F_i, R, U, D, V);

		const Real slimit = Real(0.01);
		D(0, 0) = clamp(D(0, 0), Real(slimit), Real(1 / slimit));
		D(1, 1) = clamp(D(1, 1), Real(slimit), Real(1 / slimit));
		D(2, 2) = clamp(D(2, 2), Real(slimit), Real(1 / slimit));
		//eigens[pId] = Coord(clamp(D(0, 0), Real(slimit), Real(1/ slimit)), clamp(D(1, 1), Real(slimit), Real(1 / slimit)), clamp(D(2, 2), Real(slimit), Real(1 / slimit)));
		eigens[pId] = Coord(D(0, 0), D(1, 1), D(2, 2));
		matU[pId] = U;
		matV[pId] = V;
		F[pId] = U*D*V.transpose();
		//RU[pId] = Matrix::identityMatrix();
	}

	//¼ì²éÌØÕ÷Öµ
	template <typename Real, typename Coord>
	__global__ void HM_Blend(
		DeviceArray<Real> blend,
		DeviceArray<Coord> eigens)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= blend.size()) return;

		Coord eigen_i = eigens[pId];
		Real l = min(eigen_i[0], min(eigen_i[1], eigen_i[2]));

		Real l_max = 0.8;
		Real l_min = 0.2;

		Real value = (l_max - l) / (l_max - l_min);
		blend[pId] = 0;// clamp(value, Real(0), Real(1));
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
		stressTensor[pId] = F_i * (2 * lambda * E);
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

	template <typename Real>
	__device__ Real HM_Interpolant(Real lambda)
	{
		Real l_max = 0.15;
		Real l_min = 0.05;
		return clamp((l_max - lambda) / (l_max - l_min), Real(0), Real(1));
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_JacobiLinear(
		DeviceArray<Coord> source,
		DeviceArray<Matrix> A,
		DeviceArray<Coord> y_pre,
		DeviceArray<Matrix> F,
		NeighborList<NPair> restShapes,
		DeviceArray<Real> volume,
		Real horizon,
		Real dt,
		DeviceArray<Real> fraction)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= source.size()) return;

		SmoothKernel<Real> kernSmooth;

		Matrix F_i = F[pId];
		if ((F_i.determinant()) < -0.001f)
		{
			F_i = Matrix::identityMatrix();
		}

		Real V_i = volume[pId];

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
				Matrix F_j = F[j];

				if ((F_i.determinant()) < -0.001f)
				{
					F_j = Matrix::identityMatrix();
				}

				const Real scale = V_i*V_j;

				const Real sw_ij = dt*dt*scale*weight;

				Matrix PK2_ij1 = 24000 * 2.0 * Matrix::identityMatrix();
				Matrix PK2_ij2 = 24000 * 2.0 * Matrix::identityMatrix();

				Coord rest_dir_ij = (F_i + F_j)*(y_rest_i - np_j.pos);

				rest_dir_ij = rest_dir_ij.norm() > EPSILON ? rest_dir_ij.normalize() : Coord(0, 0, 0);

				source_i += sw_ij*PK2_ij1*y_pre_j + sw_ij*PK2_ij2*r*rest_dir_ij;
				mat_i += sw_ij*PK2_ij1;

			}
		}

		source[pId] += source_i*(1 - fraction[pId]);
		A[pId] += mat_i*(1 - fraction[pId]);

	}


	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_JacobiStep(
		DeviceArray<Coord> source,
		DeviceArray<Matrix> A,
		DeviceArray<Coord> y_next,
		DeviceArray<Coord> y_pre,
		DeviceArray<Coord> y_old,
		DeviceArray<Matrix> matU,
		DeviceArray<Matrix> matV,
		DeviceArray<Coord> eigen,
		DeviceArray<Matrix> F,
		NeighborList<NPair> restShapes,
		Real horizon,
		DeviceArray<Real> volume,
		Real dt,
		EnergyType type,
		DeviceArray<Real> fraction)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		SmoothKernel<Real> kernSmooth;
		HyperelasticityModel<Real, Matrix>* model = getElasticityModel<Real, Matrix>(type);
		//StVKModel<Real, Matrix> model;

		Real lambda_i1 = eigen[pId][0];
		Real lambda_i2 = eigen[pId][1];
		Real lambda_i3 = eigen[pId][2];

		Matrix U_i = matU[pId];
		Matrix V_i = matV[pId];
		Matrix PK1_i = U_i*model->getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3)*U_i.transpose();
		Matrix PK2_i = U_i*model->getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3)*V_i.transpose();

		Real Vol_i = volume[pId];
		Matrix F_i = F[pId];

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

				Real Vol_j = volume[j];
				Matrix F_j = F[j];

				const Real scale = Vol_i*Vol_j;

				const Real sw_ij = dt*dt*scale*weight;

				Real lambda_j1 = eigen[j][0];
				Real lambda_j2 = eigen[j][1];
				Real lambda_j3 = eigen[j][2];

				Matrix U_j = matU[j];
				Matrix V_j = matV[j];
				Matrix PK1_j = U_j * model->getStressTensorPositive(lambda_j1, lambda_j2, lambda_j3)*U_j.transpose();
				Matrix PK2_j = U_j * model->getStressTensorNegative(lambda_j1, lambda_j2, lambda_j3)*V_j.transpose();
				Matrix PK1_ij = PK1_i + PK1_j;
				Matrix PK2_ij = PK2_i + PK2_j;
				
				Coord y_pre_ij = (y_pre_i - y_pre_j);

				Coord x_ij = y_rest_i - np_j.pos;
				Coord x_i = F_i*x_ij;
				Coord x_j = F_j*x_ij;
				Coord y_pre_ij_i = y_pre_ij;
				Coord y_pre_ij_j = y_pre_ij;

				Coord y_pre_j_i = y_pre_j;
				Coord y_pre_j_j = y_pre_j;

				source_i += sw_ij*PK2_i*x_ij;
 				source_i += sw_ij*PK1_i*y_pre_j_i;

				source_i += sw_ij*PK2_j*x_ij;
				source_i += sw_ij*PK1_j*y_pre_j_j;

				mat_i += sw_ij*PK1_ij;
			}
		}

		source[pId] += source_i*fraction[pId];
		A[pId] += mat_i*fraction[pId];

		delete model;
	}

		template <typename Real, typename Coord, typename Matrix>
		__global__ void HM_ComputeNextPosition(
			DeviceArray<Coord> y_next,
			DeviceArray<Coord> y_pre,
			DeviceArray<Coord> y_old,
			DeviceArray<Real> volume,
			DeviceArray<Coord> source,
			DeviceArray<Matrix> A,
			EnergyType type)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= y_next.size()) return;

			HyperelasticityModel<Real, Matrix>* model = getElasticityModel<Real, Matrix>(type);
			Real mass_i = volume[pId] * model->density;

			Matrix mat_i = A[pId] + mass_i*Matrix::identityMatrix();
			Coord src_i = source[pId] + mass_i*y_old[pId];

			y_next[pId] = mat_i.inverse()*src_i;

			delete model;
		}

	template<typename TDataType>
	bool HyperelasticityModule_test<TDataType>::initializeImpl()
	{
		m_F.resize(this->inPosition()->getElementCount());
		m_invF.resize(this->inPosition()->getElementCount());
		m_invK.resize(this->inPosition()->getElementCount());

		m_eigenValues.resize(this->inPosition()->getElementCount());
		m_matU.resize(this->inPosition()->getElementCount());
		m_matV.resize(this->inPosition()->getElementCount());
		m_volume.resize(this->inPosition()->getElementCount());

		m_energy.resize(this->inPosition()->getElementCount());
		m_gradient.resize(this->inPosition()->getElementCount());

		y_pre.resize(this->inPosition()->getElementCount());
		y_next.resize(this->inPosition()->getElementCount());
		y_current.resize(this->inPosition()->getElementCount());

		m_source.resize(this->inPosition()->getElementCount());
		m_A.resize(this->inPosition()->getElementCount());

		m_fraction.resize(this->inPosition()->getElementCount());

		m_bFixed.resize(this->inPosition()->getElementCount());
		m_fixedPos.resize(this->inPosition()->getElementCount());
		this->m_points_move_type.resize(this->inPosition()->getElementCount());

		initializeVolume();
		initializeFixed();

		m_reduce = Reduction<Real>::Create(this->inPosition()->getElementCount());

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
		int numOfParticles = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		HM_InitVolume << <pDims, BLOCK_SIZE >> > (m_volume);
	}

	template <typename Coord>
	__global__ void HM_InitFixedPos(
		DeviceArray<bool> bFixed,
		DeviceArray<Coord> fixedPos,
		DeviceArray<Coord> position
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		if (pId < FIXEDNUM || pId >= position.size()- FIXEDNUM)
		{
			bFixed[pId] = true;
			fixedPos[pId] = position[pId];
		}
		else
		{
			bFixed[pId] = false;
			fixedPos[pId] = position[pId];
		}
	}

	__global__ void HM_InitPointsMoveType(
		DeviceArray<int> m_points_move_type
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= m_points_move_type.size()) return;

		if (pId < FIXEDNUM) {
			m_points_move_type[pId] = 1;
		}
		if (pId >= m_points_move_type.size() - FIXEDNUM)
		{
			m_points_move_type[pId] = 2;
		}
		else
		{
			m_points_move_type[pId] = 0;
		}
	}

	template <typename Coord>
	__global__ void HM_AdjustFixedPos(
		DeviceArray<bool> bFixed,
		DeviceArray<Coord> fixedPos
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= fixedPos.size()) return;

		if (pId >= fixedPos.size() - FIXEDNUM)
		{
			fixedPos[pId] -= Coord(0.001, 0, 0);
		}
	}

	template <typename Coord>
	__global__ void HM_AdjustFixedPos_move(
		int adjust_type,
		DeviceArray<int> m_points_move_type,
		DeviceArray<Coord> fixedPos,
		Coord move_length,
		double x_border
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= fixedPos.size()) return;

		if (m_points_move_type[pId] == adjust_type)
		{
			fixedPos[pId] += move_length;
			//fixedPos[pId] += Coord(0.0075, 0, 0);
			if (fixedPos[pId][0] <= x_border) {
				m_points_move_type[pId] = 0;
			}
		}
	}

	template <typename Coord, typename Matrix>
	__global__ void HM_AdjustFixedPos_rotate(
		int adjust_type,
		DeviceArray<int> m_points_move_type,
		DeviceArray<Coord> fixedPos,
		Matrix rotate_mat
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= fixedPos.size()) return;

		if (m_points_move_type[pId] == adjust_type)
		{
			fixedPos[pId] = rotate_mat * fixedPos[pId];
			//fixedPos[pId] += Coord(0.0075, 0, 0);
		}
	}

	template <typename Coord>
	__global__ void HM_EnforceFixedPos(
		DeviceArray<Coord> position,
		DeviceArray<bool> bFixed,
		DeviceArray<Coord> fixedPos
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= fixedPos.size()) return;

		if (bFixed[pId])
		{
			position[pId] = fixedPos[pId];
		}
		if (pId == position.size() - FIXEDNUM - 2)
		{
			printf("Pos %f \n", position[pId][0]);
		}
	}

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::initializeFixed()
	{
		int numOfParticles = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		HM_InitFixedPos << <pDims, BLOCK_SIZE >> > (m_bFixed, m_fixedPos, this->inPosition()->getValue());
		HM_InitPointsMoveType << <pDims, BLOCK_SIZE >> > (m_points_move_type);
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

		RandNumber gen(pId);

		Matrix rotM(0);
		float theta = 3.1415926 / 4.0f;
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
		//position[pId] = origin + rotM*(position[pId] - origin);
		position[pId][0] += 0.01*(gen.Generate() - 0.5);
		position[pId][1] += 0.01*(gen.Generate() - 0.5);
		position[pId][2] += 0.01*(gen.Generate() - 0.5);
//		position[pId][1] = - position[pId][1] + 0.1;
//		position[pId][1] += (0.5 - position[pId][1]) + 0.5;

// 		if (pId >= FIXEDNUM && pId < position.size() - FIXEDNUM)
// 		{
// 			position[pId] -= Coord(0.0035, 0, 0);
// 		}
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

// 		if (pId == grad.size() - FIXEDNUM - 2)
// // 		if (grad[pId].norm() > 0.00001)
//  			printf("Thread ID %d: %f, %f, %f \n", pId, grad[pId][0], grad[pId][1], grad[pId][2]);
	}

	template <typename Real, typename Coord>
	__global__ void HM_ComputeCurrentPosition(
		DeviceArray<Coord> grad,
		DeviceArray<Coord> y_current,
		DeviceArray<Coord> y_next,
		Real alpha)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		y_next[pId] = y_current[pId] + alpha*grad[pId];
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

	template <typename Coord>
	__global__ void HM_Chebyshev_Acceleration(DeviceArray<Coord> next_X, DeviceArray<Coord> X, DeviceArray<Coord> prev_X, float omega)
	{
		int pId = blockDim.x * blockIdx.x + threadIdx.x;
		if (pId >= prev_X.size())	return;

		next_X[pId] = (next_X[pId] - X[pId])*0.666 + X[pId];

		next_X[pId] = omega*(next_X[pId] - prev_X[pId]) + prev_X[pId];
	}

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticityImplicit()
	{
		int numOfParticles = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		this->m_weights.reset();

		Log::sendMessage(Log::User, "\n \n \n \n *************solver start!!!***************");

// 		if (ind_num == 0)
// 		{
// 			HM_RotateInitPos <Coord, Matrix> << <pDims, BLOCK_SIZE >> > (this->inPosition()->getValue());
// 			ind_num++;
// 		}


		HM_AdjustFixedPos_move << <pDims, BLOCK_SIZE >> > (
			2,
			m_points_move_type,
			m_fixedPos,
			Coord(-0.0001, 0.0, 0.0),
			0.489);

		HM_EnforceFixedPos << <pDims, BLOCK_SIZE >> > (
			this->inPosition()->getValue(),
			m_bFixed,
			m_fixedPos);

		/**************************** Jacobi method ************************************************/
		// initialize y_now, y_next_iter
		Function1Pt::copy(y_pre, this->inPosition()->getValue());
		Function1Pt::copy(y_current, this->inPosition()->getValue());
		Function1Pt::copy(y_next, this->inPosition()->getValue());
		Function1Pt::copy(m_position_old, this->inPosition()->getValue());

		// do Jacobi method Loop
		bool convergeFlag = false; // converge or not
		int iterCount = 0;

		Real omega;
		while (iterCount < this->getIterationNumber()) {

			m_source.reset();
			m_A.reset();

			HM_ComputeF << <pDims, BLOCK_SIZE >> > (
				m_F,
				m_eigenValues,
				m_invK,
				m_matU,
				m_matV,
				y_current,
				this->m_restShape.getValue(),
				this->inHorizon()->getValue());
			cuSynchronize();

			HM_Blend << <pDims, BLOCK_SIZE >> > (
				m_fraction,
				m_eigenValues);

			HM_JacobiStep << <pDims, BLOCK_SIZE >> > (
				m_source,
				m_A,
				y_next,
				y_current,
				m_position_old,
				m_matU,
				m_matV,
				m_eigenValues,
				m_F,
				this->m_restShape.getValue(),
				this->inHorizon()->getValue(),
				m_volume, this->getParent()->getDt(),
				m_energyType,
				m_fraction);

			HM_JacobiLinear << <pDims, BLOCK_SIZE >> > (
				m_source,
				m_A,
				y_current,
				m_F,
				this->m_restShape.getValue(),
				m_volume,
				this->inHorizon()->getValue(),
				this->getParent()->getDt(),
				m_fraction);

			HM_ComputeNextPosition << <pDims, BLOCK_SIZE >> > (
				y_next,
				y_current,
				m_position_old,
				m_volume,
				m_source,
				m_A,
				m_energyType);

			HM_ComputeGradient << <pDims, BLOCK_SIZE >> > (
				m_gradient,
				y_current,
				y_next);
			cuSynchronize();

			HM_EnforceFixedPos << <pDims, BLOCK_SIZE >> > (
				y_next,
				m_bFixed,
				m_fixedPos);

			//stepsize adjustment
/*			Real totalE_current;
			Real totalE_next;
			getEnergy(totalE_current, y_current);
			getEnergy(totalE_next, y_next);

			printf("Current: %f Next: %f \n", totalE_current, totalE_next);

			//stepping
			
			Real alpha = 1.0f;
			int step = 0;
			while (totalE_next > totalE_current + 1.0 && step > 10)
			{
				step++;
				alpha *= 0.5;

// 				printf("Previous: %f Next: %f \n", totalE_current, totalE_next);
// 				printf("Iteration %d Step %d alpha: %f \n", iterCount, step, alpha);

				HM_ComputeCurrentPosition << <pDims, BLOCK_SIZE >> > (
					m_gradient,
					y_current,
					y_next,
					alpha);

				getEnergy(totalE_next, y_next);
			}*/
			

			/*
			int step = 0;
			if (bChebyshevAccOn)
			{
				if (step <= 10)		omega = 1;
				else if (step == 11)	omega = 2 / (2 - rho*rho);
				else	omega = 4 / (4 - rho*rho*omega);

				HM_Chebyshev_Acceleration << <pDims, BLOCK_SIZE >> > (
					y_next,
					y_current,
					y_pre,
					omega);
			}*/


			Function1Pt::copy(y_pre, y_current);
			Function1Pt::copy(y_current, y_next);

// 			Real totalE_final;
// 			getEnergy(totalE_final, y_current);

//			printf("Final: %f \n", totalE_final);

			iterCount++;
		}

		test_HM_UpdatePosition << <pDims, BLOCK_SIZE >> > (
			this->inPosition()->getValue(),
			this->inVelocity()->getValue(),
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

		//StVKModel<Real, Matrix> model;
		HyperelasticityModel<Real, Matrix>* model = getElasticityModel<Real, Matrix>(type);

		Coord eigen_i = eigens[pId];

		energy[pId] = model->getEnergy(eigen_i[0], eigen_i[1], eigen_i[2]);

		delete model;
	}

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::getEnergy(Real& totalEnergy, DeviceArray<Coord>& position)
	{
		int numOfParticles = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		HM_EnforceFixedPos << <pDims, BLOCK_SIZE >> > (
			position,
			m_bFixed,
			m_fixedPos);

		HM_ComputeF << <pDims, BLOCK_SIZE >> > (
			m_F,
			m_eigenValues,
			m_invK,
			m_matU,
			m_matV,
			position,
			this->m_restShape.getValue(),
			this->inHorizon()->getValue());

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