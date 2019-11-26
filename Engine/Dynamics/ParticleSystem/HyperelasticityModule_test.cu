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
		DeviceArray<Coord> y_new,
		DeviceArray<Coord> y_tmp,
		DeviceArray<Coord> y_old,
		DeviceArray<Matrix> stressTensor,
		DeviceArray<Matrix> F,
		DeviceArray<Matrix> invF,
		DeviceArray<Matrix> invL,
		DeviceArray<Matrix> invK,
		DeviceArray<Coord> position,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real mass,
		Real volume,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_tmp.size()) return;

		SmoothKernel<Real> kernSmooth;

		const Real scale = volume*volume;

		int size_i = restShapes.getNeighborSize(pId);

		Real mu = 48000;
		Real lambda = 12000;

		Coord y_i = y_tmp[pId];
		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		//Matrix PK_i = stressTensor[pId] * invK[pId];
		Matrix PK_i = stressTensor[pId] * invL[pId];

		Matrix mat_i(0);
		Coord mv_i = y_old[pId];
		Coord totalSource_i(0);
		Coord dy_i(0);
		Real weight_i = 0.0f;
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord y_j = y_tmp[j];
			Real r = (np_j.pos - rest_pos_i).norm();

			if (r > EPSILON)
			{
				Matrix F_j = F[j];

				Real weight = kernSmooth.Weight(r, horizon);

				//Matrix PK_j = stressTensor[j] * invL[j];
// 				Matrix PK_j = F_j*F_j.transpose()*F_j*invL[j];
// 				Matrix FL_j = F_j*invL[j];
// 
// 				Matrix PK_ij = mu*dt*dt* scale * weight * (PK_j);
// 				Matrix FL_ij = mu*dt*dt* scale * weight * (FL_j);
// 				mat_i += PK_ij;
// 
// 				//totalSource_i += PK_ij*(y_old[j]-y_old[pId]);
// 				totalSource_i += PK_ij*(y_j)+FL_ij*(y_i - y_j);

// 				Real PK_ij = mu*dt*dt*scale*weight;// *Matrix::identityMatrix();
// 				Real FL_ij = mu*dt*dt*scale*weight;
// 
// 				Coord rest_dir_ij = F_j*(rest_pos_i - np_j.pos);
// 				rest_dir_ij = rest_dir_ij.norm() > EPSILON ? rest_dir_ij.normalize() : Coord(0);
// 				totalSource_i += PK_ij*(y_j+r*rest_dir_ij);
// 				weight_i += PK_ij;

// 				Real vol_i = invL[j].determinant();
// 				Matrix PK_j = F_j*F_j.transpose();
// 
// 				Matrix R, U, D, V;
// 				polarDecomposition(PK_j, R, U, D, V);
// 
// 				D(0, 0) = pow(D(0, 0), 1.0f);
// 				D(1, 1) = pow(D(1, 1), 1.0f);
// 				D(2, 2) = pow(D(2, 2), 1.0f);
// 
// 				Matrix FL_j = F_j*invL[j];
// 				Matrix PK_ij = mu*dt*dt*scale*weight*U*D*V.transpose();
// 				Matrix FL_ij = mu*dt*dt*scale*weight*Matrix::identityMatrix();
// 
// 				Coord rest_dir_ij = /*F_j*invL[j]**/(y_i - y_j);// F_j*(rest_pos_i - np_j.pos);
// 				rest_dir_ij = rest_dir_ij.norm() > EPSILON ? rest_dir_ij.normalize() : Coord(0);
// 				totalSource_i += PK_ij*y_j + FL_ij*r*rest_dir_ij;
// 				mat_i += PK_ij;


				float PK_j = (y_i - y_j).normSquared()/(r*r);

				float PK_ij = mu*dt*dt*scale*weight*PK_j;
				float FL_ij = mu*dt*dt*scale*weight;// *pow(PK_j, 0.7f);

				Coord rest_dir_ij = /*F_j*invL[j]**/(y_i - y_j);// F_j*(rest_pos_i - np_j.pos);
				rest_dir_ij = rest_dir_ij.norm() > EPSILON ? rest_dir_ij.normalize() : Coord(0);
				totalSource_i += PK_ij*y_j + FL_ij*r*rest_dir_ij;
				weight_i += PK_ij;

				if (pId == 0)
				{
					printf("PK_ij: %f \n", PK_ij);
				}
			}
		}

		totalSource_i += mass*mv_i;

		weight_i += mass;
		y_new[pId] = totalSource_i / weight_i;

// 		totalSource_i += mass*mv_i;
// 
// 		mat_i += mass*Matrix::identityMatrix();
// 		y_new[pId] = mat_i.inverse()*totalSource_i;




		if (F[pId].determinant() < 0)
		{
			printf("Determinant_i: %e \n; Error \n; Error \n; Error \n; Error \n; Error \n",
				F[pId].determinant());
		}


		if (pId == 0)
		{
			printf("F: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
				F[pId](0, 0), F[pId](0, 1), F[pId](0, 2),
				F[pId](1, 0), F[pId](1, 1), F[pId](1, 2),
				F[pId](2, 0), F[pId](2, 1), F[pId](2, 2));
		}

// 		weight_i += 1.0f;
// 		y_new[pId] = totalSource_i / weight_i;

/*		Coord y_i = y_old[pId];
		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		//Matrix PK_i = stressTensor[pId] * invK[pId];
		Matrix PK_i = stressTensor[pId] * invL[pId];

		float mat_i(0);
		Coord mv_i = v_old[pId];
		Coord totalSource_i(0);
		Coord dy_i(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord y_j = y_old[j];
			Real r = (np_j.pos - rest_pos_i).norm();
			Real cur_r = (y_j - y_i).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);

				Matrix PK_j = stressTensor[j] * invL[j];

				Coord cur_dir_ij = y_i - y_j;
				cur_dir_ij = cur_dir_ij.norm() > EPSILON ? cur_dir_ij.normalize() : Coord(0);

				//Real PK_ij = dt*dt* scale * weight * mu * (cur_r - r) / r;
				Real PK_ij = dt*dt* scale * weight * mu;
				mat_i += PK_ij;

				//totalSource_i += PK_ij*(y_old[j]-y_old[pId]);
				totalSource_i += PK_ij*(y_old[j] + r*cur_dir_ij);
			}
		}

		if (pId == 0)
		{
			double tp_d = totalSource_i[0];
			printf("totalSource_i: \n %e %e %e \n \n",
				totalSource_i[0], totalSource_i[1], totalSource_i[2]);
		}

		totalSource_i += mv_i;

		mat_i += 1.0;
		y_new[pId] = totalSource_i / mat_i;*/



// 		if (pId == 0)
// 		{
// 			printf("tmpPK_i: \n %f %f %f \n %f %f %f \n %f %f %f \n %f	\n",
// 				mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
// 				mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
// 				mat_i(2, 0), mat_i(2, 1), mat_i(2, 2), (v_old[pId]- y_new[pId]).norm());
// 		}

/*		if ((y_new[pId]- v_old[pId]).norm() > 0.1f*horizon)
		{
// 			printf("F_i: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
// 				F[pId](0, 0), F[pId](0, 1), F[pId](0, 2),
// 				F[pId](1, 0), F[pId](1, 1), F[pId](1, 2),
// 				F[pId](2, 0), F[pId](2, 1), F[pId](2, 2));

			SmoothKernel<Real> kernSmooth;

			Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
			int size_i = restShapes.getNeighborSize(pId);

			Real total_weight = Real(0);
			Matrix matL_i(0);
			Matrix matK_i(0);
			Matrix invL_i(0);
			Matrix invK_i(0);
			Matrix F_i(0);
			Matrix invF_i(0);
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
			invK_i = V*D*U.transpose();
			F_i = matL_i*V*D*U.transpose();

			polarDecomposition(matL_i, R, U, D, V);
			D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
			D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
			D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;
			invL_i = V*D*U.transpose();
			invF_i = matK_i*V*D*U.transpose();

// 			printf("F_i: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
// 				F_i(0, 0), F_i(0, 1), F_i(0, 2),
// 				F_i(1, 0), F_i(1, 1), F_i(1, 2),
// 				F_i(2, 0), F_i(2, 1), F_i(2, 2));
// 
// 			printf("invF_i: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
// 				invF_i(0, 0), invF_i(0, 1), invF_i(0, 2),
// 				invF_i(1, 0), invF_i(1, 1), invF_i(1, 2),
// 				invF_i(2, 0), invF_i(2, 1), invF_i(2, 2));

			printf("matL_i: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
				matL_i(0, 0), matL_i(0, 1), matL_i(0, 2),
				matL_i(1, 0), matL_i(1, 1), matL_i(1, 2),
				matL_i(2, 0), matL_i(2, 1), matL_i(2, 2));

			printf("matK_i: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
				matK_i(0, 0), matK_i(0, 1), matK_i(0, 2),
				matK_i(1, 0), matK_i(1, 1), matK_i(1, 2),
				matK_i(2, 0), matK_i(2, 1), matK_i(2, 2));
		}*/

	}

	template<typename TDataType>
	bool HyperelasticityModule_test<TDataType>::initializeImpl()
	{
		m_F.resize(this->m_position.getElementCount());
		m_invK.resize(this->m_position.getElementCount());
		m_invF.resize(this->m_position.getElementCount());
		m_invL.resize(this->m_position.getElementCount());
		m_firstPiolaKirchhoffStress.resize(this->m_position.getElementCount());

		m_energy.resize(this->m_position.getElementCount());
		m_gradient.resize(this->m_position.getElementCount());

		m_reduce = Reduction<Real>::Create(this->m_position.getElementCount());

		return ElasticityModule::initializeImpl();
	}

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticity()
	{
		solveElasticityImplicit();
		//solveElasticityExplicit();
	}


	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticityExplicit()
	{
		int numOfParticles = this->m_position.getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		this->m_displacement.reset();
		this->m_weights.reset();

		Log::sendMessage(Log::User, "*************solver start!!!***************");

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
				mass, volume, 
				this->m_mu.getValue(),
				this->m_lambda.getValue(),
				this->getParent()->getDt());

			Function1Pt::copy(y_pre, y_next);

			iterCount++;
		}

		y_pre.release();
		y_next.release();
	}

	int ind_num = 0;

	template <typename Coord, typename Matrix>
	__global__ void HM_RotateInitPos(
		DeviceArray<Coord> position)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		Matrix rotM(0);
		float theta = 0.0f;// 3.1415926 / 6.0f;
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

		printf("Thread ID %d: %f, %f, %f \n", pId, grad[pId][0], grad[pId][1], grad[pId][2]);
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

		this->m_displacement.reset();
		this->m_weights.reset();

		Log::sendMessage(Log::User, "\n \n \n \n *************solver start!!!***************");

		/****************************************************************************************************/
		//-test: compute the g-inverse deformation tensor & Piola-Kirchhoff tensor


		// mass and volume are set 1.0, (need modified) 
		Real mass = 1.0;
		Real volume = 1.0;

		if (ind_num == 0)
		{
			HM_RotateInitPos <Coord, Matrix> << <pDims, BLOCK_SIZE >> > (this->m_position.getValue());
			ind_num++;
		}


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
				y_pre,
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

			for (int subIter = 0; subIter < 1; subIter++)
			{
				HM_JacobiStep << <pDims, BLOCK_SIZE >> > (
					y_next,
					y_pre,
					m_position_old,
					m_firstPiolaKirchhoffStress,
					m_F,
					m_invF,
					m_invL,
					m_invK,
					this->m_position.getValue(),
					this->m_restShape.getValue(),
					this->m_horizon.getValue(),
					mass, volume, this->getParent()->getDt());


				HM_ComputeGradient << <pDims, BLOCK_SIZE >> > (
					m_gradient,
					y_pre, 
					y_next);
				cuSynchronize();

				//stepsize adjustment
				{
					

					HM_Compute1DEnergy << <pDims, BLOCK_SIZE >> > (
						m_energy,
						m_position_old,
						y_pre,
						this->m_restShape.getValue(),
						this->m_horizon.getValue(),
						this->m_mu.getValue(),
						this->m_lambda.getValue());

					Real totalE_before = m_reduce->accumulate(m_energy.getDataPtr(), m_energy.size());

					HM_Compute1DEnergy << <pDims, BLOCK_SIZE >> > (
						m_energy,
						m_position_old,
						y_next,
						this->m_restShape.getValue(),
						this->m_horizon.getValue(),
						this->m_mu.getValue(),
						this->m_lambda.getValue());

					Real totalE_current = m_reduce->accumulate(m_energy.getDataPtr(), m_energy.size());

					Real low_limit = 0.0f;
					Real upper_limit = 1.0f;
					Real alpha = 1.0f;

					while (totalE_current > totalE_before + 100.0)
					{
						printf("alpha: %f \n", alpha);

						alpha *= 0.5;

						HM_ComputeCurrentPosition << <pDims, BLOCK_SIZE >> > (
							m_gradient,
							y_pre,
							y_next,
							alpha);

						HM_Compute1DEnergy << <pDims, BLOCK_SIZE >> > (
							m_energy,
							m_position_old,
							y_next,
							this->m_restShape.getValue(),
							this->m_horizon.getValue(),
							this->m_mu.getValue(),
							this->m_lambda.getValue());

						totalE_current = m_reduce->accumulate(m_energy.getDataPtr(), m_energy.size());
					}


					Function1Pt::copy(y_pre, y_next);
				}
			}

//			Function1Pt::copy(y_pre, y_next);

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


	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticityGradientDescent()
	{

	}


	template <typename Real, typename Matrix>
	__global__ void HM_ComputeEnergy(
		DeviceArray<Real> energy,
		DeviceArray<Matrix> F,
		Real mu,
		Real lambda)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= energy.size()) return;

		Matrix F_i = F[pId];
		Matrix E_i = F_i*F_i.transpose()- Matrix::identityMatrix();
		Real trE_i = E_i(0, 0) + E_i(1, 1) + E_i(2, 2);
		Matrix EE_i = E_i*E_i;
		Real trEE_i = EE_i(0, 0) + EE_i(1, 1) + EE_i(2, 2);

		energy[pId] = 0.5*lambda*trE_i*trE_i + mu*trEE_i;
	}

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::getEnergy(Real& totalEnergy)
	{
		int numOfParticles = this->m_position.getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		HM_ComputeFandInverse << <pDims, BLOCK_SIZE >> > (
			m_invK,
			m_invL,
			m_F,
			m_invF,
			this->m_position.getValue(),
			this->m_restShape.getValue(),
			this->m_horizon.getValue());
		cuSynchronize();

		HM_ComputeEnergy << <pDims, BLOCK_SIZE >> > (
			m_energy,
			m_F,
			m_mu.getValue(),
			m_lambda.getValue());

		totalEnergy = m_reduce->accumulate(m_energy.getDataPtr(), m_energy.size());
	}


#ifdef PRECISION_FLOAT
	template class HyperelasticityModule_test<DataType3f>;
#else
	template class HyperelasticityModule_test<DataType3d>;
#endif
}