#include "HyperelasticityModule_NewtonMethod.h"
#include "Core/Utility.h"
#include "Framework/Framework/Node.h"
#include "Core/Algorithm/MatrixFunc.h"
#include "Kernel.h"

#include "Framework/Framework/Log.h"
#include "Core/Utility/Function1Pt.h"
#include "Core/Utility/math_utilities.h"

namespace Physika
{
	template <typename Coord>
	COMM_FUNC bool isExitNaN_vec3f(Coord vec) {
		float tmp = vec[0] + vec[1] + vec[2];
		if (isnan(tmp))return true;
		else return false;
	}

	template <typename Matrix>
	COMM_FUNC bool isExitNaN_mat3f(Matrix mat) {
		float tmp = mat(0, 0) + mat(0, 1) + mat(0, 2) + mat(1, 0)
			+ mat(1, 1) + mat(1, 2) + mat(2, 0) + mat(2, 1) + mat(2, 2);
		if (isnan(tmp))return true;
		else return false;
	}

	template <typename Real, typename Coord>
	__global__ void computeDelta_vec(
		DeviceArray<Coord> vec1,
		DeviceArray<Coord> vec2,
		DeviceArray<Real> delta_norm)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vec1.size()) return;

		delta_norm[pId] = (vec1[pId] - vec2[pId]).norm();
	}

	template <typename Real, typename Coord>
	__global__ void computeNorm_vec(
		DeviceArray<Coord> vec,
		DeviceArray<Real> norm)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vec.size()) return;

		norm[pId] = vec[pId].norm();
	}

	template <typename Real, typename Matrix>
	__global__ void compute_mat_sum(
		DeviceArray<Matrix> mats,
		DeviceArray<Real> matItem_sum)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mats.size()) return;

		Matrix mat = mats[pId];
		Real sum = mat(0, 0) + mat(0, 1) + mat(0, 2)
			+ mat(1, 0) + mat(1, 1) + mat(1, 2)
			+ mat(2, 0) + mat(2, 1) + mat(2, 2);
		matItem_sum[pId] = sum;
	}

	template <typename Real, typename Matrix>
	__global__ void compute_mat_norm(
		DeviceArray<Matrix> mats,
		DeviceArray<Real> matItem_sum)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mats.size()) return;

		Matrix mat = mats[pId];
		Real sum = mat.frobeniusNorm();
		matItem_sum[pId] = sum;
	}

	template<typename TDataType>
	HyperelasticityModule_NewtonMethod<TDataType>::HyperelasticityModule_NewtonMethod()
		: ElasticityModule<TDataType>()
		, m_energyType(Linear)
	{
	}

	template <typename Coord, typename Matrix>
	COMM_FUNC Coord vec3_dot_mat3(
		Coord vec,
		Matrix mat) 
	{
		Coord result;
		result[0] = vec[0] * mat(0, 0) + vec[1] * mat(1, 0) + vec[2] * mat(2, 0);
		result[1] = vec[0] * mat(0, 1) + vec[1] * mat(1, 1) + vec[2] * mat(2, 1);
		result[2] = vec[0] * mat(0, 2) + vec[1] * mat(1, 2) + vec[2] * mat(2, 2);
		return result;
	}
	template <typename Coord, typename Matrix>
	COMM_FUNC Matrix vec3_outer_product_vec3(
		Coord vec1,
		Coord vec2,
		Matrix mat)
	{
		Matrix result;
		result(0, 0) += vec1[0] * vec2[0] ; result(0, 1) += vec1[0] * vec2[1] ; result(0, 2) += vec1[0] * vec2[2] ;
		result(1, 0) += vec1[1] * vec2[0] ; result(1, 1) += vec1[1] * vec2[1] ; result(1, 2) += vec1[1] * vec2[2] ;
		result(2, 0) += vec1[2] * vec2[0] ; result(2, 1) += vec1[2] * vec2[1] ; result(2, 2) += vec1[2] * vec2[2] ;
		return result;
	}
	template <typename Real, typename Matrix>
	COMM_FUNC Real mat3_double_product_mat3(
		Matrix mat1,
		Matrix mat2,
		Real type_arg)
	{
		return mat1(0, 0)*mat2(0, 0) + mat1(0, 1)*mat2(0, 1) + mat1(0, 2)*mat2(0, 2)
			+ mat1(1, 0)*mat2(1, 0) + mat1(1, 1)*mat2(1, 1) + mat1(1, 2)*mat2(1, 2)
			+ mat1(2, 0)*mat2(2, 0) + mat1(2, 1)*mat2(2, 1) + mat1(2, 2)*mat2(2, 2);
	}


	//**********compute total weight of each particle************************
	template <typename Real, typename Coord, typename NPair>
	__global__ void HM_ComputeTotalWeight_newton(
		DeviceArray<Coord> position,
		NeighborList<NPair> restShapes,
		DeviceArray<Real> totalWeight,
		Real horizon)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		SmoothKernel<Real> kernSmooth;

		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		int size_i = restShapes.getNeighborSize(pId);

		Real total_weight = Real(0);
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				total_weight += weight;
			}
		}

		totalWeight[pId] = total_weight;

	}

	// *************************  only update position **************************
	template <typename Coord>
	__global__ void HM_UpdatePosition_only(
		DeviceArray<Coord> position,
		DeviceArray<Coord> y_next)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		position[pId] = y_next[pId];
	}

	template <typename Coord>
	__global__ void HM_UpdatePosition_delta_only(
		DeviceArray<Coord> position,
		DeviceArray<Coord> delta_y)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		position[pId] = position[pId] + delta_y[pId];
	}

	template <typename Coord>
	__global__ void HM_UpdatePosition_Velocity(
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

	template <typename Coord>
	__global__ void HM_UpdateVelocity_only(
		DeviceArray<Coord> position,
		DeviceArray<Coord> velocity,
		DeviceArray<Coord> position_old,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		velocity[pId] += (position[pId] - position_old[pId]) / dt;
	}


	template <typename Real, typename Coord, typename Matrix>
	__global__ void HM_ComputeTotalEnergy_Linear(
		DeviceArray<Real> energy_momentum,
		DeviceArray<Real> energy_elseticity,
		DeviceArray<Coord> position,
		DeviceArray<Coord> position_old,
		DeviceArray<Matrix> F,
		Real mu,
		Real lambda,
		Real mass,
		Real volume,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		Matrix F_i = F[pId];
		Matrix epsilon = 0.5*(F_i + F_i.transpose()) - Matrix::identityMatrix();
		Real elasticity_energy_density_i = mu * mat3_double_product_mat3(epsilon, epsilon, mass) + 0.5*lambda*epsilon.trace()*epsilon.trace();

		energy_momentum[pId] = 0.5*mass * (position[pId] - position_old[pId]).normSquared() / (dt*dt);
		energy_elseticity[pId] = volume* elasticity_energy_density_i;
	}
	template <typename Real, typename Coord, typename Matrix>
	__global__ void HM_ComputeTotalEnergy_StVK(
		DeviceArray<Real> energy_momentum,
		DeviceArray<Real> energy_elseticity,
		DeviceArray<Coord> position,
		DeviceArray<Coord> position_old,
		DeviceArray<Matrix> F,
		Real mass,
		Real volume,
		Real mu,
		Real lambda,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		Matrix F_i = F[pId];
		Matrix E_i = 0.5*(F_i.transpose()*F_i - Matrix::identityMatrix());
		Real elasticity_energy_density_i = mu * mat3_double_product_mat3(E_i, E_i, mass) + 0.5*lambda*E_i.trace()*E_i.trace();

		energy_momentum[pId] = 0.5*mass * (position[pId] - position_old[pId]).normSquared() / (dt*dt);
		energy_elseticity[pId] = volume * elasticity_energy_density_i;
	}


	template <typename Real, typename Coord, typename Matrix>
	COMM_FUNC Matrix HM_ComputeHessianItem_LinearEnergy(
		int index_energy_k,
		int index_i,
		int index_j,
		Coord dx_ik,
		Coord dx_jk,
		Coord delta_x_i,
		Coord delta_x_j,
		Real horizon,
		Real mu, Real lambda,
		Real mass, Real volume,
		Real weight_ki,
		Real weight_kj,
		Matrix identityMat)
	{
		if (index_energy_k == index_i) {
			// k == i == j
			if (index_i == index_j) {
				Matrix result(0.0);
				result = volume * volume * (
					mu*(delta_x_i.dot(delta_x_i))*Matrix::identityMatrix() 
					+(mu + lambda)*vec3_outer_product_vec3(delta_x_i, delta_x_i, Matrix::identityMatrix() ));
				return result;
			}
			// k == i != j
			else {
				Matrix result(0.0);
				
				result = weight_kj * volume*volume*( 
					mu*dx_jk.dot(delta_x_i)*Matrix::identityMatrix() 
					+ mu*vec3_outer_product_vec3(dx_jk, delta_x_i, Matrix::identityMatrix())
					+ lambda*vec3_outer_product_vec3(delta_x_i, dx_jk, Matrix::identityMatrix()) );
		
				return result;
			}
		}
		else{
			Matrix result(0.0);
			//i==j, k!=i
			if (index_i == index_j) {
				result = weight_ki * weight_ki * volume*volume*(
					mu*dx_ik.dot(dx_ik)*Matrix::identityMatrix()
					+ (mu + lambda) * vec3_outer_product_vec3(dx_ik, dx_ik, Matrix::identityMatrix()) );
				
				return result;
			}
			// k == j, i!=j
			else if(index_energy_k == index_j){
				result = weight_ki * volume*volume*(
					mu*delta_x_j.dot(dx_ik)*Matrix::identityMatrix()
					+ mu * vec3_outer_product_vec3(delta_x_j, dx_ik, Matrix::identityMatrix())
					+ lambda * vec3_outer_product_vec3(dx_ik, delta_x_j, Matrix::identityMatrix()));
			
				return result;
			}
			// k != i != j 
			else {
				result = weight_ki * weight_kj * volume*volume*(
					mu*dx_jk.dot(dx_ik)*Matrix::identityMatrix()
					+ mu * vec3_outer_product_vec3(dx_jk, dx_ik, Matrix::identityMatrix())
					+ lambda * vec3_outer_product_vec3(dx_ik, dx_jk, Matrix::identityMatrix()));

				return result;
			}
		}
		
		
	}


	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_ComputeSourceTerm(
		DeviceArray<Coord> sourceItems,
		DeviceArray<Matrix> inverseK,
		DeviceArray<Matrix> stressTensors,
		DeviceArray<Coord> position_old,
		DeviceArray<Coord> y_current,
		DeviceArray<Coord> Sum_delta_x,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real mu, Real lambda,
		Real mass, Real volume, Real dt,
		Real weightScale) 
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= stressTensors.size()) return;

		Coord delta_x_i = Sum_delta_x[pId];
		Coord y_i = y_current[pId];
		Matrix invK_i = inverseK[pId];
		int index_i = pId;
		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		int size_i = restShapes.getNeighborSize(pId);

		Coord energy_gradient_i = Coord(0.0);
		energy_gradient_i += mass * (y_current[pId] - position_old[pId]) / (dt*dt);
		Coord linear_gradient_Wi_i = volume * (stressTensors[index_i]*delta_x_i) ;
		energy_gradient_i += volume * linear_gradient_Wi_i; // not finished

		Coord b_i = Coord(0.0);

		SmoothKernel<Real> kernSmooth;

		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int index_j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			Coord y_j = y_current[index_j];

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				weight = weight / weightScale;

				Matrix invK_j = inverseK[index_j];
				Coord delta_x_j = Sum_delta_x[index_j];
				Coord dx_ji = vec3_dot_mat3((rest_pos_j - rest_pos_i) / (horizon*horizon), invK_i);
				Coord dx_ij = vec3_dot_mat3((rest_pos_i - rest_pos_j) / (horizon*horizon), invK_j);
				
				Coord linear_gradient_Wj_i = weight * volume *(stressTensors[index_j] * dx_ij);
				energy_gradient_i += volume * linear_gradient_Wj_i;
			}
		}
		b_i = -energy_gradient_i;

		sourceItems[pId] = b_i;
	}

	template <typename Real, typename Coord, typename Matrix>
	COMM_FUNC Matrix HM_ComputeHessianItem_StVKEnergy(
		int index_energy_k,
		int index_i,
		int index_j,
		Coord dx_ik,
		Coord dx_jk,
		Coord delta_x_i,
		Coord delta_x_j,
		Real horizon,
		Real mu, Real lambda,
		Real mass, Real volume,
		Real weight_ki,
		Real weight_kj,
		Matrix F_k,
		Matrix E_k)
	{
		if (index_energy_k == index_i) {
			// k == i == j
			if (index_i == index_j) {
				Matrix result(0.0);
				result = volume * volume * (
					2*mu*(delta_x_i.dot( E_k*delta_x_i ))*Matrix::identityMatrix()
					+ mu*vec3_outer_product_vec3(F_k * delta_x_i, F_k * delta_x_i, Matrix::identityMatrix()) 
					+ mu * delta_x_i.dot(delta_x_i)*( F_k * F_k.transpose() )
					+ lambda* vec3_outer_product_vec3(F_k * delta_x_i, F_k * delta_x_i, Matrix::identityMatrix())
					+ lambda * E_k.trace() * delta_x_i.dot(delta_x_i) * Matrix::identityMatrix()
					);
				return result;
			}
			// k == i != j
			else {
				Matrix result(0.0);

				result = weight_kj * volume * volume * (
					2 * mu*(dx_jk.dot(E_k*delta_x_i))*Matrix::identityMatrix()
					+ mu * vec3_outer_product_vec3(F_k * dx_jk, F_k * delta_x_i, Matrix::identityMatrix())
					+ mu * dx_jk.dot(delta_x_i)*(F_k * F_k.transpose())
					+ lambda * vec3_outer_product_vec3(F_k * delta_x_i, F_k * dx_jk, Matrix::identityMatrix())
					+ lambda * E_k.trace() * dx_jk.dot(delta_x_i) * Matrix::identityMatrix()
					);

				return result;
			}
		}
		else{
			Matrix result(0.0);
			// i == j, k != i
			if (index_i == index_j) {
				result = weight_ki * weight_ki * volume * volume * (
					2 * mu*(dx_ik.dot(E_k*dx_ik))*Matrix::identityMatrix()
					+ mu * vec3_outer_product_vec3(F_k * dx_ik, F_k * dx_ik, Matrix::identityMatrix())
					+ mu * dx_ik.dot(dx_ik)*(F_k * F_k.transpose())
					+ lambda * vec3_outer_product_vec3(F_k * dx_ik, F_k * dx_ik, Matrix::identityMatrix())
					+ lambda * E_k.trace() * dx_ik.dot(dx_ik) * Matrix::identityMatrix()
					);

				return result;
			}
			// k == j, k != i
			else if(index_energy_k == index_j){
				result = weight_ki * volume * volume * (
					2 * mu*(delta_x_j.dot(E_k*dx_ik))*Matrix::identityMatrix()
					+ mu * vec3_outer_product_vec3(F_k * delta_x_j, F_k * dx_ik, Matrix::identityMatrix())
					+ mu * delta_x_j.dot(dx_ik)*(F_k * F_k.transpose())
					+ lambda * vec3_outer_product_vec3(F_k * dx_ik, F_k * delta_x_j, Matrix::identityMatrix())
					+ lambda * E_k.trace() * delta_x_j.dot(dx_ik) * Matrix::identityMatrix()
					);

				return result;
			}
			// k != i, k != j
			else {
				result = weight_ki * weight_kj * volume * volume * (
					2 * mu*(dx_jk.dot(E_k*dx_ik))*Matrix::identityMatrix()
					+ mu * vec3_outer_product_vec3(F_k * dx_jk, F_k * dx_ik, Matrix::identityMatrix())
					+ mu * dx_jk.dot(dx_ik)*(F_k * F_k.transpose())
					+ lambda * vec3_outer_product_vec3(F_k * dx_ik, F_k * dx_jk, Matrix::identityMatrix())
					+ lambda * E_k.trace() * dx_jk.dot(dx_ik) * Matrix::identityMatrix()
					);

				return result;
			}
		}
	}

	// these deformation gradients are mat3x3, may be singular
	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_ComputeFandSdx(
		DeviceArray<Matrix> inverseK,
		DeviceArray<Matrix> F,
		DeviceArray<Coord> Sum_delta_x,
		DeviceArray<Coord> position,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real weightScale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		SmoothKernel<Real> kernSmooth;

		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		int size_i = restShapes.getNeighborSize(pId);

		Matrix matL_i(0);
		Matrix matK_i(0);
		Coord Delta_x = Coord(0.0);

		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				weight = weight / weightScale;

				Coord p = (position[j] - position[pId]) / horizon;
				Coord q = (rest_pos_j - rest_pos_i) / horizon;

				Delta_x += weight * (rest_pos_i - rest_pos_j)/(horizon*horizon);

				matL_i(0, 0) += p[0] * q[0] * weight; matL_i(0, 1) += p[0] * q[1] * weight; matL_i(0, 2) += p[0] * q[2] * weight;
				matL_i(1, 0) += p[1] * q[0] * weight; matL_i(1, 1) += p[1] * q[1] * weight; matL_i(1, 2) += p[1] * q[2] * weight;
				matL_i(2, 0) += p[2] * q[0] * weight; matL_i(2, 1) += p[2] * q[1] * weight; matL_i(2, 2) += p[2] * q[2] * weight;

				matK_i(0, 0) += q[0] * q[0] * weight; matK_i(0, 1) += q[0] * q[1] * weight; matK_i(0, 2) += q[0] * q[2] * weight;
				matK_i(1, 0) += q[1] * q[0] * weight; matK_i(1, 1) += q[1] * q[1] * weight; matK_i(1, 2) += q[1] * q[2] * weight;
				matK_i(2, 0) += q[2] * q[0] * weight; matK_i(2, 1) += q[2] * q[1] * weight; matK_i(2, 2) += q[2] * q[2] * weight;

			}
		}

		/*Matrix R, U, D, V;
		polarDecomposition(matK_i, R, U, D, V);
		Real threshold = 0.0001f*horizon;
		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;*/

		Matrix inv_mat_K = matK_i.inverse();

		inverseK[pId] = inv_mat_K;

		Delta_x = vec3_dot_mat3(Delta_x, inv_mat_K);
		Sum_delta_x[pId] = Delta_x;
		F[pId] = matL_i * inv_mat_K;
	}

	template <typename Real, typename Matrix>
	__global__ void HM_ComputeFirstPiolaKirchhoff_Linear(
		DeviceArray<Matrix> stressTensor,
		DeviceArray<Matrix> F,
		Real mu,
		Real lambda)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= F.size()) return;

		Matrix F_i = F[pId];

		// find infinitesimal strain tensor epsilon = 1/2(F + F^T) - I
		Matrix epsilon = 0.5*(F_i.transpose() + F_i) - Matrix::identityMatrix();
		// find first Piola-Kirchhoff matix; Linear material
		stressTensor[pId] = 2 * mu * epsilon + lambda * epsilon.trace() * Matrix::identityMatrix();

	}

	template <typename Real, typename Matrix>
	__global__ void HM_ComputeFirstPiolaKirchhoff_StVK(
		DeviceArray<Matrix> stressTensor,
		DeviceArray<Matrix> F,
		Real mu,
		Real lambda)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= F.size()) return;

		Matrix F_i = F[pId];

		// find strain tensor E = 1/2(F^T * F - I)
		Matrix E = 0.5*(F_i.transpose() * F_i - Matrix::identityMatrix());
		// find first Piola-Kirchhoff matix; StVK material
		stressTensor[pId] = F_i * (2 * mu * E + lambda * E.trace() * Matrix::identityMatrix());
	}


	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_ComputeHessianMatrix_Linear(
		DeviceArray<Matrix> inverseK,
		DeviceArray<Coord> Sum_delta_x,
		NeighborList<NPair> restShapes,
		NeighborList<int> hessian_query_list,
		DeviceArray<int> hessian_matrix_index,
		DeviceArray<Matrix> m_hessian_matrix,
		Real horizon,
		Real mu, Real lambda,
		Real mass,
		Real volume,
		Real dt,
		Real weightScale) 
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= inverseK.size()) return;

		Coord delta_x_i = Sum_delta_x[pId];

		Matrix invK_i = inverseK[pId];

		int index_i = pId;
		int neighbor_size_i = restShapes.getNeighborSize(index_i);
		Coord rest_pos_i = restShapes.getElement(index_i, 0).pos;
		int hessian_size_i = hessian_query_list.getNeighborSize(index_i);

		SmoothKernel<Real> kernSmooth;
		for (int hessian_i = 0; hessian_i < hessian_size_i; hessian_i++)
		{
			int index_j = hessian_query_list.getElement(index_i, hessian_i);
			int hessian_matrix_index_ij = hessian_matrix_index[index_i] + hessian_i;
			NPair np_j = restShapes.getElement(index_j, 0);
			Coord rest_pos_j = np_j.pos;
			
			Coord delta_x_j = Sum_delta_x[index_j];

			Matrix hessian_i_j = Matrix(0.0);

			// compute H_ij , i and k in N(i): H_ij = Wi_i_j + sum of Wk_i_j
			for (int neighbor_i = 0; neighbor_i < neighbor_size_i; neighbor_i++) {
				NPair np_k = restShapes.getElement(index_i, neighbor_i);
				int index_k = np_k.index;
				Coord rest_pos_k = np_k.pos;

				Real r_ki = (rest_pos_k - rest_pos_i).norm();
				Real r_kj = (rest_pos_k - rest_pos_j).norm();

				// some conditions H_ij will be zero
				if (r_kj >= horizon) { continue; }
				if (r_kj <= EPSILON && index_k != index_j) { continue; }
				if (r_ki <= EPSILON && index_k != index_i) { continue; }

				Real weight_ki, weight_kj;
				Coord dx_jk, dx_ik;
				Matrix invK_k = inverseK[index_k];
				Real volume_k = volume;

				if (index_k == index_j) {weight_kj = 0.0;	dx_jk = Coord(0.0);}
				else {
					weight_kj = kernSmooth.Weight(r_kj, horizon);
					weight_kj = weight_kj / weightScale;
					dx_jk = vec3_dot_mat3((rest_pos_j - rest_pos_k) / (horizon*horizon), invK_k);
				}
				if (index_k == index_i) {weight_ki = 0.0;	dx_ik = Coord(0.0);}
				else {
					weight_ki = kernSmooth.Weight(r_ki, horizon);
					weight_ki = weight_ki / weightScale;
					dx_ik = vec3_dot_mat3((rest_pos_i - rest_pos_k) / (horizon*horizon), invK_k);
				}
				
				Matrix partial_Wk_i_j = HM_ComputeHessianItem_LinearEnergy(
					index_k, index_i, index_j,
					dx_ik, dx_jk,
					delta_x_i, delta_x_j,
					horizon,
					mu, lambda,
					mass, volume,
					weight_ki, weight_kj,
					Matrix::identityMatrix());
				hessian_i_j += volume_k * partial_Wk_i_j;
			}	// i & N(i) traversed
			if (index_i == index_j) { 
				hessian_i_j += mass*Matrix::identityMatrix() /(dt*dt); 
				Matrix H_ii_inverse = hessian_i_j.inverse();
				bool isH_ii_inverse_Nan = isExitNaN_mat3f(H_ii_inverse);
				if (isH_ii_inverse_Nan) { printf("Nan First Inverse"); }
			}
			m_hessian_matrix[hessian_matrix_index_ij] = hessian_i_j;

			//printf("hessian matrix item:%d, %d : %f\n",index_i,index_j, hessian_i_j.frobeniusNorm());
		}

	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_ComputeHessianMatrix_StVK(
		DeviceArray<Matrix> m_F,
		DeviceArray<Matrix> inverseK,
		DeviceArray<Coord> Sum_delta_x,
		NeighborList<NPair> restShapes,
		NeighborList<int> hessian_query_list,
		DeviceArray<int> hessian_matrix_index,
		DeviceArray<Matrix> m_hessian_matrix,
		Real horizon,
		Real mu, Real lambda,
		Real mass,
		Real volume,
		Real dt,
		Real weightScale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= inverseK.size()) return;

		Coord delta_x_i = Sum_delta_x[pId];

		Matrix invK_i = inverseK[pId];

		int index_i = pId;
		int neighbor_size_i = restShapes.getNeighborSize(index_i);
		Coord rest_pos_i = restShapes.getElement(index_i, 0).pos;
		int hessian_size_i = hessian_query_list.getNeighborSize(index_i);

		SmoothKernel<Real> kernSmooth;
		for (int hessian_i = 0; hessian_i < hessian_size_i; hessian_i++)
		{
			int index_j = hessian_query_list.getElement(index_i, hessian_i);
			int hessian_matrix_index_ij = hessian_matrix_index[index_i] + hessian_i;
			NPair np_j = restShapes.getElement(index_j, 0);
			Coord rest_pos_j = np_j.pos;

			Coord delta_x_j = Sum_delta_x[index_j];

			Matrix hessian_i_j = Matrix(0.0);

			// compute H_ij , i and k in N(i): H_ij = Wi_i_j + sum of Wk_i_j
			for (int neighbor_i = 0; neighbor_i < neighbor_size_i; neighbor_i++) {
				NPair np_k = restShapes.getElement(index_i, neighbor_i);
				int index_k = np_k.index;
				Coord rest_pos_k = np_k.pos;

				Real r_ki = (rest_pos_k - rest_pos_i).norm();
				Real r_kj = (rest_pos_k - rest_pos_j).norm();

				// some conditions H_ij will be zero
				if (r_kj >= horizon) { continue; }
				if (r_kj <= EPSILON && index_k != index_j) { continue; }
				if (r_ki <= EPSILON && index_k != index_i) { continue; }

				Real weight_ki, weight_kj;
				Coord dx_jk, dx_ik;
				Matrix invK_k = inverseK[index_k];
				Real volume_k = volume;

				if (index_k == index_j) { weight_kj = 0.0;	dx_jk = Coord(0.0); }
				else {
					weight_kj = kernSmooth.Weight(r_kj, horizon);
					weight_kj = weight_kj / weightScale;
					dx_jk = vec3_dot_mat3((rest_pos_j - rest_pos_k) / (horizon*horizon), invK_k);
				}
				if (index_k == index_i) { weight_ki = 0.0;	dx_ik = Coord(0.0); }
				else {
					weight_ki = kernSmooth.Weight(r_ki, horizon);
					weight_ki = weight_ki / weightScale;
					dx_ik = vec3_dot_mat3((rest_pos_i - rest_pos_k) / (horizon*horizon), invK_k);
				}

				Matrix F_k = m_F[index_k];
				Matrix E_k = 0.5*(F_k.transpose() * F_k - Matrix::identityMatrix());
				Matrix partial_Wk_i_j = HM_ComputeHessianItem_StVKEnergy(
					index_k, index_i, index_j,
					dx_ik, dx_jk,
					delta_x_i, delta_x_j,
					horizon,
					mu, lambda,
					mass, volume,
					weight_ki, weight_kj,
					F_k, E_k);
				hessian_i_j += volume_k * partial_Wk_i_j;
			}	// i & N(i) traversed
			if (index_i == index_j) {
				hessian_i_j += mass * Matrix::identityMatrix() / (dt*dt);
				Matrix H_ii_inverse = hessian_i_j.inverse();
				bool isH_ii_inverse_Nan = isExitNaN_mat3f(H_ii_inverse);
				if (isH_ii_inverse_Nan) { printf("Nan First Inverse"); }
			}
			m_hessian_matrix[hessian_matrix_index_ij] = hessian_i_j;

			//printf("hessian matrix item:%d, %d : %f\n", index_i, index_j, hessian_i_j.frobeniusNorm());
		}

	}

	template <typename Coord, typename Matrix>
	__global__ void HM_JacobiStep(
		DeviceArray<Coord> delta_y_new,
		DeviceArray<Coord> delta_y_old,
		DeviceArray<Coord> sourceItems,
		NeighborList<int> hessian_query_list,
		DeviceArray<int> hessian_matrix_index,
		DeviceArray<Matrix> m_hessian_matrix)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= delta_y_old.size()) return;

		Coord totalSource_i = sourceItems[pId];
		// not finished

		int index_i = pId;
		int hessian_size_i = hessian_query_list.getNeighborSize(index_i);
		Matrix H_ii = Matrix::identityMatrix();

		for (int hessian_i = 0; hessian_i < hessian_size_i; hessian_i++)
		{
			int index_j = hessian_query_list.getElement(index_i, hessian_i);
			int hessian_matrix_index_ij = hessian_matrix_index[index_i] + hessian_i;
			if (index_j == index_i) {
				H_ii = m_hessian_matrix[hessian_matrix_index_ij];
			}
			else {
				Matrix H_ij = m_hessian_matrix[hessian_matrix_index_ij];
				Coord delta_y_j = delta_y_old[index_j];
				totalSource_i -= H_ij * delta_y_j;
			}
		}

		Matrix H_ii_inverse = H_ii.inverse();
		delta_y_new[pId] = H_ii_inverse * totalSource_i;
	}

	template<typename TDataType>
	bool HyperelasticityModule_NewtonMethod<TDataType>::initializeImpl()
	{
		m_position_old.resize(this->m_position.getElementCount());
		m_F.resize(this->m_position.getElementCount());
		m_invK.resize(this->m_position.getElementCount());
		m_firstPiolaKirchhoffStress.resize(this->m_position.getElementCount());

		m_totalWeight.resize(this->m_position.getElementCount());
		m_Sum_delta_x.resize(this->m_position.getElementCount());
		m_source_items.resize(this->m_position.getElementCount() );

		this->m_position.connect(this->hessian_query.m_position);
		this->hessian_query.setRadius(2*this->m_horizon.getValue());
		printf("horizon: %f\n", this->m_horizon.getValue());
		this->hessian_query.initialize();
		this->m_hessian_matrix.resize(this->hessian_query.getNeighborList().getElements().size());
		printf("hessian sparse matrix size: %d\n", this->m_hessian_matrix.size());

		return ElasticityModule::initializeImpl();
	}


	template<typename TDataType>
	void HyperelasticityModule_NewtonMethod<TDataType>::solveElasticity()
	{
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;


		int numOfParticles = this->m_position.getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);
		HM_ComputeTotalWeight_newton << <pDims, BLOCK_SIZE >> > (
			this->m_position.getValue(),
			this->m_restShape.getValue(),
			this->m_totalWeight,
			this->m_horizon.getValue());
		cuSynchronize();
		{
			Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
			Real max_totalWeight = pReduction->maximum(this->m_totalWeight.getDataPtr(), numOfParticles);
			printf("Max total weight: %f \n", max_totalWeight);
		}

		solveElasticity_NewtonMethod_StVK();

	}

	template<typename TDataType>
	void HyperelasticityModule_NewtonMethod<TDataType>::solveElasticity_NewtonMethod()
	{
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		int numOfParticles = this->m_position.getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		this->m_displacement.reset();
		this->m_weights.reset();

		Log::sendMessage(Log::User, "solver start!!!");

		// mass and volume are set 1.0, (need modified) 
		Real mass = 1.0;
		Real volume = 1.0;

		// initialize y_now, y_next_iter
		DeviceArray<Coord> delta_y_pre(numOfParticles);
		DeviceArray<Coord> delta_y_next(numOfParticles);

		delta_y_pre.reset();
		delta_y_next.reset();
		Function1Pt::copy(m_position_old, this->m_position.getValue());

		// do Jacobi method Loop
		bool newton_convergeFlag = false; // outer loop(newton method) converge or not
		bool jacobi_convergeFlag = false; // inner loop(jacobi method) converge or not
		int newton_iteNum = 0;
		int jacobi_iteNum = 0;
		int jacobi_total_iteNum = 0;
		int	newton_maxIterations = 50;
		int jacobi_maxIterations = 200;
		double converge_threshold = 0.001f*this->m_horizon.getValue();
		double newton_relative_error_threshold = 0.001;
		double jacobi_relative_error_threshold = 0.000001;

		double newton_first_delta = 0.0;
		double jacobi_first_delta = 0.0;

		double last_state_energy_momentum = 0.0;
		double last_state_energy_elasticity = 0.0;

		int energy_rise_times = 0;

		HM_ComputeHessianMatrix_Linear << <pDims, BLOCK_SIZE >> > (
			m_invK,
			m_Sum_delta_x,
			this->m_restShape.getValue(),
			this->hessian_query.getNeighborList(),
			this->hessian_query.getNeighborList().getIndex(),
			this->m_hessian_matrix,
			this->m_horizon.getValue(),
			this->m_mu.getValue(),
			this->m_lambda.getValue(),
			mass, volume, this->getParent()->getDt(),
			this->weightScale);
		cuSynchronize();

		HM_ComputeFandSdx << <pDims, BLOCK_SIZE >> > (
			m_invK,
			m_F,
			m_Sum_delta_x,
			this->m_position.getValue(),
			this->m_restShape.getValue(),
			this->m_horizon.getValue(),
			this->weightScale);
		cuSynchronize();

		{	//Debug: energy
			DeviceArray<Real> energy_momentum(numOfParticles);
			DeviceArray<Real> energy_elasticity(numOfParticles);
			HM_ComputeTotalEnergy_Linear << <pDims, BLOCK_SIZE >> > (
				energy_momentum,
				energy_elasticity,
				this->m_position.getValue(),
				m_position_old,
				m_F,
				this->m_mu.getValue(),
				this->m_lambda.getValue(),
				mass, volume,
				this->getParent()->getDt());
			cuSynchronize();

			Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
			Real current_energy_momentum = pReduction->accumulate(energy_momentum.getDataPtr(), numOfParticles);
			Real current_energy_elasticity = pReduction->accumulate(energy_elasticity.getDataPtr(), numOfParticles);
			energy_momentum.release();
			energy_elasticity.release();

			last_state_energy_momentum = current_energy_momentum;
			last_state_energy_elasticity = current_energy_elasticity;
			printf("energy:%e, \t %e\n", current_energy_momentum, current_energy_elasticity);
		}
		Real newton_first_delta_energy(0.0);

		for (newton_iteNum = 0; newton_iteNum < newton_maxIterations; ++newton_iteNum) { // newton method loop: H*y_{k+1} = H*y_{k} + gradient of f 

			delta_y_pre.reset();
			delta_y_next.reset();

			HM_ComputeFirstPiolaKirchhoff_Linear << <pDims, BLOCK_SIZE >> > (
				m_firstPiolaKirchhoffStress,
				m_F,
				this->m_mu.getValue(),
				this->m_lambda.getValue());
			cuSynchronize();

			HM_ComputeSourceTerm << <pDims, BLOCK_SIZE >> > (
				m_source_items,
				m_invK,
				m_firstPiolaKirchhoffStress,
				m_position_old,
				this->m_position.getValue(),
				m_Sum_delta_x,
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				this->m_mu.getValue(),
				this->m_lambda.getValue(),
				mass, volume, this->getParent()->getDt(),
				this->weightScale);
			cuSynchronize();

			if (is_debug)
			{	// Debug: source item
				DeviceArray<Real> sourceItem_norm(numOfParticles);
				computeNorm_vec << <pDims, BLOCK_SIZE >> >(m_source_items, sourceItem_norm);
				cuSynchronize();

				Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
				Real max_sourceItem = pReduction->maximum(sourceItem_norm.getDataPtr(), numOfParticles);
				Real ave_sourceItem = pReduction->average(sourceItem_norm.getDataPtr(), numOfParticles);
				sourceItem_norm.release();

				printf("max & average source item: %e, %e\n", max_sourceItem, ave_sourceItem);
			}

			jacobi_convergeFlag = false;

			for (jacobi_iteNum = 0; jacobi_iteNum < jacobi_maxIterations; ++jacobi_iteNum) { // jacobi method loop

				HM_JacobiStep << <pDims, BLOCK_SIZE >> > (
					delta_y_next,
					delta_y_pre,
					m_source_items,
					this->hessian_query.getNeighborList(),
					this->hessian_query.getNeighborList().getIndex(),
					this->m_hessian_matrix);
				cuSynchronize();

				if(is_debug)
				{ // compute jacobi converge
					Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
					DeviceArray<Real> Delta_y_norm(numOfParticles);
					computeDelta_vec << <pDims, BLOCK_SIZE >> >(delta_y_next, delta_y_pre, Delta_y_norm);
					cuSynchronize();

					Real jacobi_max_delta = pReduction->maximum(Delta_y_norm.getDataPtr(), numOfParticles);
					Delta_y_norm.release();

					//printf("%f ", jacobi_max_delta);
					if (jacobi_iteNum == 0) {
						jacobi_first_delta = jacobi_max_delta;
						if (jacobi_first_delta == 0.0 ) { jacobi_convergeFlag = true; }
					}
					else {
						if ( (jacobi_max_delta /jacobi_first_delta) < jacobi_relative_error_threshold )
						{ 
							jacobi_convergeFlag = true; 
						}
					}
				}

				Function1Pt::copy(delta_y_pre, delta_y_next);
				if (jacobi_convergeFlag) { break; }
			}

			if (jacobi_iteNum < jacobi_maxIterations) { jacobi_iteNum++; }
			jacobi_total_iteNum += jacobi_iteNum;

			if(is_debug)
			{	// compute newton converge
				Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
				DeviceArray<Real> Delta_y_norm(numOfParticles);

				computeNorm_vec << <pDims, BLOCK_SIZE >> >(delta_y_next, Delta_y_norm);
				cuSynchronize();

				Real max_delta = pReduction->maximum(Delta_y_norm.getDataPtr(), numOfParticles);
				Delta_y_norm.release();
				printf("max Delta pos: %.e\n", max_delta);

				if (newton_iteNum == 0) {
					newton_first_delta = max_delta;
					if (newton_first_delta == 0.0) { newton_convergeFlag = true; }
				}
				else {
					if ( (max_delta /newton_first_delta) < newton_relative_error_threshold) { newton_convergeFlag = true; }
				}
			}

			HM_UpdatePosition_delta_only << <pDims, BLOCK_SIZE >> > (
				this->m_position.getValue(),
				delta_y_next);
			cuSynchronize();

			HM_ComputeFandSdx << <pDims, BLOCK_SIZE >> > (
				m_invK,
				m_F,
				m_Sum_delta_x,
				this->m_position.getValue(),
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				this->weightScale);
			cuSynchronize();

			{	//Debug: energy
				DeviceArray<Real> energy_momentum(numOfParticles);
				DeviceArray<Real> energy_elasticity(numOfParticles);
				HM_ComputeTotalEnergy_Linear << <pDims, BLOCK_SIZE >> > (
					energy_momentum,
					energy_elasticity,
					this->m_position.getValue(),
					m_position_old,
					m_F,
					this->m_mu.getValue(),
					this->m_lambda.getValue(),
					mass, volume,
					this->getParent()->getDt());
				cuSynchronize();

				Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
				Real current_energy_momentum = pReduction->accumulate(energy_momentum.getDataPtr(), numOfParticles);
				Real current_energy_elasticity = pReduction->accumulate(energy_elasticity.getDataPtr(), numOfParticles);
				energy_momentum.release();
				energy_elasticity.release();

				Real delta_energy = (current_energy_momentum + current_energy_elasticity)
					- (last_state_energy_momentum + last_state_energy_elasticity);
				if (delta_energy >= 0
					&& (current_energy_momentum + current_energy_elasticity)>EPSILON)
				{
					energy_rise_times++;
				}
				if (newton_iteNum == 0) {
					newton_first_delta_energy = abs(delta_energy);
					if (newton_first_delta_energy == 0.0) { newton_convergeFlag = true; }
				}
				else {
					if (abs(delta_energy) / newton_first_delta_energy < newton_relative_error_threshold) {
						newton_convergeFlag = true;
						printf("relative error:%e\n", abs(delta_energy) / newton_first_delta_energy);
					}

					printf("delta energy:%e\n", delta_energy);
				}
				
				last_state_energy_momentum = current_energy_momentum;
				last_state_energy_elasticity = current_energy_elasticity;
				printf("energy:%e, \t %e\n", current_energy_momentum, current_energy_elasticity);
			}

			if (newton_convergeFlag) { break; }
		}

		HM_UpdateVelocity_only << <pDims, BLOCK_SIZE >> > (
			this->m_position.getValue(),
			this->m_velocity.getValue(),
			m_position_old,
			this->getParent()->getDt());
		cuSynchronize();

		delta_y_pre.release();
		delta_y_next.release();

		if (newton_iteNum < newton_maxIterations) { newton_iteNum++; }
		printf("newton ite num: %d \n jacobi ave_ite num: %f \n", newton_iteNum, double(jacobi_total_iteNum) / double(newton_iteNum));
		printf("energy rise times: %d\n", energy_rise_times);
		printf("momentum energy:%e, \t elasticity energy:%e\n", last_state_energy_momentum, last_state_energy_elasticity);
		if (jacobi_convergeFlag) { printf("jacobi converge!"); }
		if (newton_convergeFlag) { printf("newton converge!"); }
	}

	template<typename TDataType>
	void HyperelasticityModule_NewtonMethod<TDataType>::solveElasticity_NewtonMethod_StVK()
	{
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		int numOfParticles = this->m_position.getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		this->m_displacement.reset();
		this->m_weights.reset();

		Log::sendMessage(Log::User, "solver start!!!");

		// mass and volume are set 1.0, (need modified) 
		Real mass = 1.0;
		Real volume = 1.0;

		// initialize y_now, y_next_iter
		DeviceArray<Coord> delta_y_pre(numOfParticles);
		DeviceArray<Coord> delta_y_next(numOfParticles);

		delta_y_pre.reset();
		delta_y_next.reset();
		Function1Pt::copy(m_position_old, this->m_position.getValue());

		// do Jacobi method Loop
		bool newton_convergeFlag = false; // outer loop(newton method) converge or not
		bool jacobi_convergeFlag = false; // inner loop(jacobi method) converge or not
		int newton_iteNum = 0;
		int jacobi_iteNum = 0;
		int jacobi_total_iteNum = 0;
		int	newton_maxIterations = 50;
		int jacobi_maxIterations = 200;
		double converge_threshold = 0.001f*this->m_horizon.getValue();
		double relative_error_threshold = 0.001;
		double newton_relative_error_threshold = 0.000001;
		double jacobi_relative_error_threshold = 0.000001;

		Real newton_first_delta = 0.0;
		Real jacobi_first_delta = 0.0;

		Real initial_energy_momentum = 0.0;
		Real initial_energy_elasticity = 0.0;
		Real last_state_energy_momentum = 0.0;
		Real last_state_energy_elasticity = 0.0;

		int energy_rise_times = 0;
		double newton_first_delta_energy = 0.0;
		Real *momentum_energy_newtonItes = new Real[newton_maxIterations + 1];
		Real *elasticity_energy_newtonItes = new Real[newton_maxIterations + 1];

		HM_ComputeFandSdx << <pDims, BLOCK_SIZE >> > (
			m_invK,
			m_F,
			m_Sum_delta_x,
			this->m_position.getValue(),
			this->m_restShape.getValue(),
			this->m_horizon.getValue(),
			this->weightScale);
		cuSynchronize();

		{	//compute initial energy
			DeviceArray<Real> energy_momentum(numOfParticles);
			DeviceArray<Real> energy_elasticity(numOfParticles);
			HM_ComputeTotalEnergy_Linear << <pDims, BLOCK_SIZE >> > (
				energy_momentum,
				energy_elasticity,
				this->m_position.getValue(),
				m_position_old,
				m_F,
				this->m_mu.getValue(),
				this->m_lambda.getValue(),
				mass, volume,
				this->getParent()->getDt());
			cuSynchronize();

			Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
			Real current_energy_momentum = pReduction->accumulate(energy_momentum.getDataPtr(), numOfParticles);
			Real current_energy_elasticity = pReduction->accumulate(energy_elasticity.getDataPtr(), numOfParticles);
			energy_momentum.release();
			energy_elasticity.release();

			initial_energy_momentum = current_energy_momentum;
			initial_energy_elasticity = current_energy_elasticity;
			last_state_energy_momentum = current_energy_momentum;
			last_state_energy_elasticity = current_energy_elasticity;
			printf("initial energy:%e, \t %e\n", current_energy_momentum, current_energy_elasticity);
		}

		// newton method loop
		for (newton_iteNum = 0; newton_iteNum < newton_maxIterations; ++newton_iteNum) {

			delta_y_pre.reset();
			delta_y_next.reset();

			HM_ComputeFirstPiolaKirchhoff_StVK << <pDims, BLOCK_SIZE >> > (
				m_firstPiolaKirchhoffStress,
				m_F,
				this->m_mu.getValue(),
				this->m_lambda.getValue());
			cuSynchronize();

			HM_ComputeSourceTerm << <pDims, BLOCK_SIZE >> > (
				m_source_items,
				m_invK,
				m_firstPiolaKirchhoffStress,
				m_position_old,
				this->m_position.getValue(),
				m_Sum_delta_x,
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				this->m_mu.getValue(),
				this->m_lambda.getValue(),
				mass, volume, this->getParent()->getDt(),
				this->weightScale);
			cuSynchronize();

			if (is_debug)
			{	// Debug: source item
				DeviceArray<Real> sourceItem_norm(numOfParticles);
				computeNorm_vec << <pDims, BLOCK_SIZE >> >(m_source_items, sourceItem_norm);
				cuSynchronize();

				Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
				Real max_sourceItem = pReduction->maximum(sourceItem_norm.getDataPtr(), numOfParticles);
				Real ave_sourceItem = pReduction->average(sourceItem_norm.getDataPtr(), numOfParticles);
				sourceItem_norm.release();

				printf("--max & average source item: %f, %f\n", max_sourceItem, ave_sourceItem);
			}

			HM_ComputeHessianMatrix_StVK << <pDims, BLOCK_SIZE >> > (
				m_F,
				m_invK,
				m_Sum_delta_x,
				this->m_restShape.getValue(),
				this->hessian_query.getNeighborList(),
				this->hessian_query.getNeighborList().getIndex(),
				this->m_hessian_matrix,
				this->m_horizon.getValue(),
				this->m_mu.getValue(),
				this->m_lambda.getValue(),
				mass, volume, this->getParent()->getDt(),
				this->weightScale);
			cuSynchronize();

			if(is_debug)
			{	// Debug: hessian matrix
				DeviceArray<Real> hessianMatrixItem_norm(this->m_hessian_matrix.size());
				compute_mat_norm << <pDims, BLOCK_SIZE >> >(this->m_hessian_matrix, hessianMatrixItem_norm);
				cuSynchronize();

				Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(this->m_hessian_matrix.size());
				Real max_hessian_norm = pReduction->maximum(hessianMatrixItem_norm.getDataPtr(), this->m_hessian_matrix.size());
				Real ave_hessian_norm = pReduction->average(hessianMatrixItem_norm.getDataPtr(), this->m_hessian_matrix.size());
				hessianMatrixItem_norm.release();

				printf("--max & average hessian item norm: %e, %e\n", max_hessian_norm, ave_hessian_norm);
			}

			jacobi_convergeFlag = false;
			for (jacobi_iteNum = 0; jacobi_iteNum < jacobi_maxIterations; ++jacobi_iteNum) { // jacobi method loop

				HM_JacobiStep << <pDims, BLOCK_SIZE >> > (
					delta_y_next,
					delta_y_pre,
					m_source_items,
					this->hessian_query.getNeighborList(),
					this->hessian_query.getNeighborList().getIndex(),
					this->m_hessian_matrix);
				cuSynchronize();

				{ // compute jacobi converge
					Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
					DeviceArray<Real> Delta_y_norm(numOfParticles);
					computeDelta_vec << <pDims, BLOCK_SIZE >> >(delta_y_next, delta_y_pre, Delta_y_norm);
					cuSynchronize();

					Real jacobi_max_delta = pReduction->maximum(Delta_y_norm.getDataPtr(), numOfParticles);
					Delta_y_norm.release();

					//printf("%f ", jacobi_max_delta);
					if (jacobi_iteNum == 0) {
						jacobi_first_delta = jacobi_max_delta;
						if (jacobi_first_delta < 1e-9) { jacobi_convergeFlag = true; }
					}
					else {
						if ((jacobi_max_delta / jacobi_first_delta) < relative_error_threshold
							|| jacobi_max_delta<1e-9)
						{
							jacobi_convergeFlag = true;
						}
					}
				}

				Function1Pt::copy(delta_y_pre, delta_y_next);
				if (jacobi_convergeFlag) { break; }
			}
			// jacobi method loop end

			if (jacobi_iteNum < jacobi_maxIterations) { jacobi_iteNum++; }
			jacobi_total_iteNum += jacobi_iteNum;

			if(this->is_debug)
			{	// compute newton converge
				Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
				DeviceArray<Real> Delta_y_norm(numOfParticles);

				computeNorm_vec << <pDims, BLOCK_SIZE >> >(delta_y_next, Delta_y_norm);
				cuSynchronize();

				Real max_delta = pReduction->maximum(Delta_y_norm.getDataPtr(), numOfParticles);
				Delta_y_norm.release();
				printf("--max Delta pos: %e\n", max_delta);
			}

			// update position only
			HM_UpdatePosition_delta_only << <pDims, BLOCK_SIZE >> > (
				this->m_position.getValue(),
				delta_y_next);
			cuSynchronize();

			// update F matrix
			HM_ComputeFandSdx << <pDims, BLOCK_SIZE >> > (
				m_invK,
				m_F,
				m_Sum_delta_x,
				this->m_position.getValue(),
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				this->weightScale);
			cuSynchronize();

			{	// update energy
				DeviceArray<Real> energy_momentum(numOfParticles);
				DeviceArray<Real> energy_elasticity(numOfParticles);
				HM_ComputeTotalEnergy_Linear << <pDims, BLOCK_SIZE >> > (
					energy_momentum,
					energy_elasticity,
					this->m_position.getValue(),
					m_position_old,
					m_F,
					this->m_mu.getValue(),
					this->m_lambda.getValue(),
					mass, volume,
					this->getParent()->getDt());
				cuSynchronize();

				Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
				Real current_energy_momentum = pReduction->accumulate(energy_momentum.getDataPtr(), numOfParticles);
				Real current_energy_elasticity = pReduction->accumulate(energy_elasticity.getDataPtr(), numOfParticles);
				energy_momentum.release();
				energy_elasticity.release();

				Real delta_energy = (current_energy_momentum - last_state_energy_momentum)
					+ (current_energy_elasticity - last_state_energy_elasticity);
				if (delta_energy >= 0.0
					&& (current_energy_momentum + current_energy_elasticity)>EPSILON)
				{
					energy_rise_times++;
				}

				double relative_error;
				if ((last_state_energy_momentum + last_state_energy_elasticity) == 0.0) {
					relative_error = 0.0;
				}
				else {
					relative_error = abs(delta_energy) / (last_state_energy_momentum + last_state_energy_elasticity);
				}
				if (relative_error < newton_relative_error_threshold) {
					newton_convergeFlag = true;
				}

				last_state_energy_momentum = current_energy_momentum;
				last_state_energy_elasticity = current_energy_elasticity;
				momentum_energy_newtonItes[newton_iteNum] = current_energy_momentum;
				elasticity_energy_newtonItes[newton_iteNum] = current_energy_elasticity;

				printf("--energy:%e, \t %e\n", current_energy_momentum, current_energy_elasticity);
				printf("--delta energy:%e\n", delta_energy);
				printf("--%dth iteration, relative error:%e\n", newton_iteNum, relative_error);
			}

			if (newton_convergeFlag) { break; }
		}

		HM_UpdateVelocity_only << <pDims, BLOCK_SIZE >> > (
			this->m_position.getValue(),
			this->m_velocity.getValue(),
			m_position_old,
			this->getParent()->getDt());
		cuSynchronize();

		delta_y_pre.release();
		delta_y_next.release();

		delete momentum_energy_newtonItes;
		delete elasticity_energy_newtonItes;

		if (newton_iteNum < newton_maxIterations) { newton_iteNum++; }
		printf("--newton ite num: %d \n jacobi ave_ite num: %f \n", newton_iteNum, double(jacobi_total_iteNum) / double(newton_iteNum));
		printf("--energy rise times: %d\n", energy_rise_times);
		printf("--momentum energy:%e, \t elasticity energy:%e\n", last_state_energy_momentum, last_state_energy_elasticity);
		if (jacobi_convergeFlag) { printf("--jacobi converge!"); }
		if (newton_convergeFlag) { printf("--newton converge!"); }
	}

#ifdef PRECISION_FLOAT
	template class HyperelasticityModule_NewtonMethod<DataType3f>;
#else
	template class HyperelasticityModule_NewtonMethod<DataType3d>;
#endif
}