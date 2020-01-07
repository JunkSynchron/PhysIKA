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
	__global__ void computeRelativeError_vec(
		DeviceArray<Coord> vec1,
		DeviceArray<Coord> vec2,
		DeviceArray<Real> relative_error)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vec1.size()) return;

		relative_error[pId] = abs( 1 - vec1[pId].norm / vec2[pId].norm );
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

	// the final result will stored in the element: array[0];
	// this function will change elements' value
	template <typename Array_T>
	void sumOfArray(Array_T array, int array_size) {

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

		source[pId] = mass * (position[pId]);
	}

	template <typename Real, typename Coord, typename Matrix>
	COMM_FUNC Matrix HM_ComputeHessianMatrix_LinearEnergy(
		int index_energy,
		int index_i,
		int index_j,
		Coord dx_ij,
		Coord dx_ji,
		Coord delta_x_i,
		Coord delta_x_j,
		Real horizon,
		Real mu, Real lambda,
		Real mass, Real volume,
		Real weight_ij,
		Real weight_ji,
		Matrix identityMat)
	{
		if (index_energy == index_i) {
			if (index_i == index_j) {
				Matrix result(0.0);
				result = volume * volume * (
					mu*(delta_x_i.dot(delta_x_i))*Matrix::identityMatrix() 
					+(mu + lambda)*vec3_outer_product_vec3(delta_x_i, delta_x_i, Matrix::identityMatrix() ));
				return result;
			}
			else {
				Matrix result(0.0);
				
				result = weight_ij * volume*volume*( 
					mu*dx_ji.dot(delta_x_i)*Matrix::identityMatrix() 
					+ mu*vec3_outer_product_vec3(dx_ji, delta_x_i, Matrix::identityMatrix())
					+ lambda*vec3_outer_product_vec3(delta_x_i, dx_ji, Matrix::identityMatrix()) );
		
				return result;
			}
		}
		else if(index_energy == index_j){
			Matrix result(0.0);

			if (index_i == index_j) {
				result = weight_ji * weight_ji * volume*volume*(
					mu*dx_ij.dot(dx_ij)*Matrix::identityMatrix()
					+ (mu + lambda) * vec3_outer_product_vec3(dx_ij, dx_ij, Matrix::identityMatrix()) );
				
				return result;
			}
			else {
				result = weight_ji * volume*volume*(
					mu*delta_x_j.dot(dx_ij)*Matrix::identityMatrix()
					+ mu * vec3_outer_product_vec3(delta_x_j, dx_ij, Matrix::identityMatrix())
					+ lambda * vec3_outer_product_vec3(dx_ij, delta_x_j, Matrix::identityMatrix()));
			
				return result;
			}
		}
		else {
			return Matrix(0.0);
		}
		
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_ComputeSourceTerm_Linear_singleEnergy(
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

		Coord delta_x = Sum_delta_x[pId];
		Coord y = y_current[pId];
		Matrix invK = inverseK[pId];

		Coord b = Coord(0.0);
		b -= mass * (y_current[pId] - position_old[pId]) / (dt*dt);
		b -= volume* volume * (stressTensors[pId] * delta_x);
		b += mass * y / (dt*dt);
		b += volume * volume*( mu* (delta_x.dot(delta_x))*y + (mu+lambda)*delta_x*(delta_x.dot(y)) );
		

		SmoothKernel<Real> kernSmooth;

		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		int size_i = restShapes.getNeighborSize(pId);

		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			Coord y_j = y_current[j];

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				weight = weight / weightScale;

				Coord dx_ji = vec3_dot_mat3((rest_pos_j - rest_pos_i) / (horizon*horizon), invK);

				b += weight*volume*volume*(mu*(dx_ji.dot(delta_x))*y_j + mu * dx_ji*(delta_x.dot(y_j)) + lambda*delta_x*(dx_ji.dot(y_j)) );
			}
		}

		sourceItems[pId] = b;
	}


	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_ComputeSourceTerm_Linear(
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

		Matrix partial_Wi_i_i = HM_ComputeHessianMatrix_LinearEnergy(
			index_i, index_i, index_i,
			Coord(0.0), Coord(0.0),
			delta_x_i, delta_x_i,
			horizon,
			mu, lambda,
			mass, volume,
			Real(0.0), Real(0.0),
			Matrix::identityMatrix() );
		Matrix hessian_i_i = (mass / (dt*dt)) * Matrix::identityMatrix() + volume*partial_Wi_i_i;
		//hessian_i_i not finnished

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
				
				Coord linear_gradient_Wj_i = weight * (stressTensors[index_j] * dx_ij);
				energy_gradient_i += volume * linear_gradient_Wj_i;

				Matrix hessian_Wj_i_i = HM_ComputeHessianMatrix_LinearEnergy(
					index_j, index_i, index_i,
					Coord(0.0), Coord(0.0),
					delta_x_i, delta_x_i,
					horizon,
					mu, lambda,
					mass, volume,
					weight, weight,
					Matrix::identityMatrix() );

				hessian_i_i += volume * hessian_Wj_i_i;

				Matrix partial_Wi_i_j = HM_ComputeHessianMatrix_LinearEnergy(
					index_i, index_i, index_j,
					dx_ij, dx_ji,
					delta_x_i, delta_x_j,
					horizon,
					mu, lambda,
					mass, volume,
					weight, weight,
					Matrix::identityMatrix());
				Matrix partial_Wj_i_j = HM_ComputeHessianMatrix_LinearEnergy(
					index_j, index_i, index_j,
					dx_ij, dx_ji,
					delta_x_i, delta_x_j,
					horizon,
					mu, lambda,
					mass, volume,
					weight, weight,
					Matrix::identityMatrix());

				b_i += (volume*partial_Wi_i_j + volume*partial_Wj_i_j)*y_j;
			}
		}

		b_i += hessian_i_i * y_i;
		b_i -= energy_gradient_i;

		sourceItems[pId] = b_i;
	}

	template <typename Real, typename Coord, typename Matrix>
	COMM_FUNC Matrix HM_ComputeHessianMatrix_StVKEnergy(
		int index_energy,
		int index_i,
		int index_j,
		Coord dx_ij,
		Coord dx_ji,
		Coord delta_x_i,
		Coord delta_x_j,
		Real horizon,
		Real mu, Real lambda,
		Real mass, Real volume,
		Real weight_ij,
		Real weight_ji,
		Matrix F,
		Matrix E)
	{
		if (index_energy == index_i) {
			if (index_i == index_j) {
				Matrix result(0.0);
				result = volume * volume * (
					2*mu*(delta_x_i.dot( E*delta_x_i ))*Matrix::identityMatrix()
					+ mu*vec3_outer_product_vec3(F * delta_x_i, F * delta_x_i, Matrix::identityMatrix()) 
					+ mu * delta_x_i.dot(delta_x_i)*( F * F.transpose() )
					+ lambda* vec3_outer_product_vec3(F * delta_x_i, F * delta_x_i, Matrix::identityMatrix())
					+ lambda * E.trace() * delta_x_i.dot(delta_x_i) * Matrix::identityMatrix()
					);
				return result;
			}
			else {
				Matrix result(0.0);

				result = weight_ij * volume * volume * (
					2 * mu*(dx_ji.dot(E*delta_x_i))*Matrix::identityMatrix()
					+ mu * vec3_outer_product_vec3(F * dx_ji, F * delta_x_i, Matrix::identityMatrix())
					+ mu * dx_ji.dot(delta_x_i)*(F * F.transpose())
					+ lambda * vec3_outer_product_vec3(F * delta_x_i, F * dx_ji, Matrix::identityMatrix())
					+ lambda * E.trace() * dx_ji.dot(delta_x_i) * Matrix::identityMatrix()
					);

				return result;
			}
		}
		else if (index_energy == index_j) {
			Matrix result(0.0);

			if (index_i == index_j) {
				result = weight_ji * weight_ji * volume * volume * (
					2 * mu*(dx_ij.dot(E*dx_ij))*Matrix::identityMatrix()
					+ mu * vec3_outer_product_vec3(F * dx_ij, F * dx_ij, Matrix::identityMatrix())
					+ mu * dx_ij.dot(dx_ij)*(F * F.transpose())
					+ lambda * vec3_outer_product_vec3(F * dx_ij, F * dx_ij, Matrix::identityMatrix())
					+ lambda * E.trace() * dx_ij.dot(dx_ij) * Matrix::identityMatrix()
					);

				return result;
			}
			else {
				result = weight_ji * volume * volume * (
					2 * mu*(delta_x_j.dot(E*dx_ij))*Matrix::identityMatrix()
					+ mu * vec3_outer_product_vec3(F * delta_x_j, F * dx_ij, Matrix::identityMatrix())
					+ mu * delta_x_j.dot(dx_ij)*(F * F.transpose())
					+ lambda * vec3_outer_product_vec3(F * dx_ij, F * delta_x_j, Matrix::identityMatrix())
					+ lambda * E.trace() * delta_x_j.dot(dx_ij) * Matrix::identityMatrix()
					);

				return result;
			}
		}
		else {
			return Matrix(0.0);
		}

	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_ComputeSourceTerm_StVK(
		DeviceArray<Coord> sourceItems,
		DeviceArray<Matrix> F,
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

		Matrix F_i = F[pId];
		Matrix E_i = 0.5*(F_i.transpose() * F_i - Matrix::identityMatrix() );
		Matrix partial_Wi_i_i = HM_ComputeHessianMatrix_StVKEnergy(
			index_i, index_i, index_i,
			Coord(0.0), Coord(0.0),
			delta_x_i, delta_x_i,
			horizon,
			mu, lambda,
			mass, volume,
			Real(0.0), Real(0.0),
			F_i, E_i);
		Matrix hessian_i_i = (mass / (dt*dt)) * Matrix::identityMatrix() + volume * partial_Wi_i_i;
		//hessian_i_i not finnished

		Coord energy_gradient_i = Coord(0.0);
		energy_gradient_i += mass * (y_current[pId] - position_old[pId]) / (dt*dt);
		Coord linear_gradient_Wi_i = volume * (stressTensors[index_i] * delta_x_i);
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

				Coord linear_gradient_Wj_i = weight * (stressTensors[index_j] * dx_ij);
				energy_gradient_i += volume * linear_gradient_Wj_i;

				Matrix F_j = F[index_j];
				Matrix E_j = 0.5*(F_j.transpose() * F_j - Matrix::identityMatrix());
				Matrix hessian_Wj_i_i = HM_ComputeHessianMatrix_StVKEnergy(
					index_j, index_i, index_i,
					Coord(0.0), Coord(0.0),
					delta_x_i, delta_x_i,
					horizon,
					mu, lambda,
					mass, volume,
					weight, weight,
					F_j, E_j);

				hessian_i_i += volume * hessian_Wj_i_i;

				Matrix partial_Wi_i_j = HM_ComputeHessianMatrix_StVKEnergy(
					index_i, index_i, index_j,
					dx_ij, dx_ji,
					delta_x_i, delta_x_j,
					horizon,
					mu, lambda,
					mass, volume,
					weight, weight,
					F_i, E_i);
				Matrix partial_Wj_i_j = HM_ComputeHessianMatrix_StVKEnergy(
					index_j, index_i, index_j,
					dx_ij, dx_ji,
					delta_x_i, delta_x_j,
					horizon,
					mu, lambda,
					mass, volume,
					weight, weight,
					F_j, E_j);

				b_i += (volume*partial_Wi_i_j + volume * partial_Wj_i_j)*y_j;
			}
		}

		b_i += hessian_i_i * y_i;
		b_i -= energy_gradient_i;

		sourceItems[pId] = b_i;
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

		Matrix R, U, D, V;
		polarDecomposition(matK_i, R, U, D, V);
		//	getSVDmatrix(matK_i, &U, &D, &V);

		Real threshold = 0.0001f*horizon;
		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;

		Matrix inv_mat_K = V * D*U.transpose();

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
	__global__ void HM_JacobiStep_Linear(
		DeviceArray<Coord> y_new,
		DeviceArray<Coord> y_old,
		DeviceArray<Coord> sourceItems,
		DeviceArray<Matrix> inverseK,
		DeviceArray<Coord> Sum_delta_x,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real mu, Real lambda,
		Real mass,
		Real volume,
		Real dt,
		Real weightScale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_old.size()) return;

		Coord totalSource_i = sourceItems[pId];
		// not finished

		Coord delta_x_i = Sum_delta_x[pId];
		Coord y_i = y_old[pId];
		Matrix invK_i = inverseK[pId];
		int index_i = pId;
		int size_i = restShapes.getNeighborSize(pId);
		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;

		Matrix partial_Wi_i_i = HM_ComputeHessianMatrix_LinearEnergy(
			index_i, index_i, index_i,
			Coord(0.0), Coord(0.0),
			delta_x_i, delta_x_i,
			horizon,
			mu, lambda,
			mass, volume,
			Real(0.0), Real(0.0),
			Matrix::identityMatrix());
		Matrix hessian_i_i = (mass / (dt*dt)) * Matrix::identityMatrix() + volume * partial_Wi_i_i;
		// hessian_i_i not finished

		SmoothKernel<Real> kernSmooth;
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int index_j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Coord y_j = y_old[index_j];
			Real r = (rest_pos_j - rest_pos_i).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				weight = weight / weightScale;

				Matrix invK_j = inverseK[index_j];
				Coord delta_x_j = Sum_delta_x[index_j];
				Coord dx_ji = vec3_dot_mat3((rest_pos_j - rest_pos_i) / (horizon*horizon), invK_i);
				Coord dx_ij = vec3_dot_mat3((rest_pos_i - rest_pos_j) / (horizon*horizon), invK_j);

				Matrix hessian_Wj_i_i = HM_ComputeHessianMatrix_LinearEnergy(
					index_j, index_i, index_i,
					Coord(0.0), Coord(0.0),
					delta_x_i, delta_x_i,
					horizon,
					mu, lambda,
					mass, volume,
					weight, weight,
					Matrix::identityMatrix());

				hessian_i_i += volume * hessian_Wj_i_i;

				Matrix partial_Wi_i_j = HM_ComputeHessianMatrix_LinearEnergy(
					index_i, index_i, index_j,
					dx_ij, dx_ji,
					delta_x_i, delta_x_j,
					horizon,
					mu, lambda,
					mass, volume,
					weight, weight,
					Matrix::identityMatrix());
				Matrix partial_Wj_i_j = HM_ComputeHessianMatrix_LinearEnergy(
					index_j, index_i, index_j,
					dx_ij, dx_ji,
					delta_x_i, delta_x_j,
					horizon,
					mu, lambda,
					mass, volume,
					weight, weight,
					Matrix::identityMatrix());

				totalSource_i -= (volume*partial_Wi_i_j + volume * partial_Wj_i_j)*y_j;
			}
		}

		y_new[pId] = hessian_i_i.inverse()*totalSource_i;
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_JacobiStep_StVK(
		DeviceArray<Coord> y_new,
		DeviceArray<Coord> y_old,
		DeviceArray<Coord> sourceItems,
		DeviceArray<Matrix> F,
		DeviceArray<Matrix> inverseK,
		DeviceArray<Coord> Sum_delta_x,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real mu, Real lambda,
		Real mass,
		Real volume,
		Real dt,
		Real weightScale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_old.size()) return;

		Coord totalSource_i = sourceItems[pId];
		// not finished

		Coord delta_x_i = Sum_delta_x[pId];
		Coord y_i = y_old[pId];
		Matrix invK_i = inverseK[pId];
		int index_i = pId;
		int size_i = restShapes.getNeighborSize(pId);
		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;

		Matrix F_i = F[pId];
		Matrix E_i = 0.5*(F_i.transpose() * F_i - Matrix::identityMatrix());
		Matrix partial_Wi_i_i = HM_ComputeHessianMatrix_StVKEnergy(
			index_i, index_i, index_i,
			Coord(0.0), Coord(0.0),
			delta_x_i, delta_x_i,
			horizon,
			mu, lambda,
			mass, volume,
			Real(0.0), Real(0.0),
			F_i, E_i);
		Matrix hessian_i_i = (mass / (dt*dt)) * Matrix::identityMatrix() + volume * partial_Wi_i_i;
		// hessian_i_i not finished

		SmoothKernel<Real> kernSmooth;
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int index_j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Coord y_j = y_old[index_j];
			Real r = (rest_pos_j - rest_pos_i).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				weight = weight / weightScale;

				Matrix invK_j = inverseK[index_j];
				Coord delta_x_j = Sum_delta_x[index_j];
				Coord dx_ji = vec3_dot_mat3((rest_pos_j - rest_pos_i) / (horizon*horizon), invK_i);
				Coord dx_ij = vec3_dot_mat3((rest_pos_i - rest_pos_j) / (horizon*horizon), invK_j);

				Matrix F_j = F[index_j];
				Matrix E_j = 0.5*(F_j.transpose() * F_j - Matrix::identityMatrix());
				Matrix hessian_Wj_i_i = HM_ComputeHessianMatrix_StVKEnergy(
					index_j, index_i, index_i,
					Coord(0.0), Coord(0.0),
					delta_x_i, delta_x_i,
					horizon,
					mu, lambda,
					mass, volume,
					weight, weight,
					F_j, E_j);

				hessian_i_i += volume * hessian_Wj_i_i;

				Matrix partial_Wi_i_j = HM_ComputeHessianMatrix_StVKEnergy(
					index_i, index_i, index_j,
					dx_ij, dx_ji,
					delta_x_i, delta_x_j,
					horizon,
					mu, lambda,
					mass, volume,
					weight, weight,
					F_i, E_i);
				Matrix partial_Wj_i_j = HM_ComputeHessianMatrix_StVKEnergy(
					index_j, index_i, index_j,
					dx_ij, dx_ji,
					delta_x_i, delta_x_j,
					horizon,
					mu, lambda,
					mass, volume,
					weight, weight,
					F_j, E_j);

				totalSource_i -= (volume*partial_Wi_i_j + volume * partial_Wj_i_j)*y_j;
			}
		}

		y_new[pId] = hessian_i_i.inverse()*totalSource_i;
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

		debug_pos_isNaN = false;
		debug_v_isNaN = false;
		debug_invL_isNaN = false;
		debug_F_isNaN = false;
		debug_invF_isNaN = false;
		debug_Piola_isNaN = true;

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

		solveElasticity_NewtonMethod();

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
		DeviceArray<Coord> y_pre(numOfParticles);
		DeviceArray<Coord> y_next(numOfParticles);

		Function1Pt::copy(y_pre, this->m_position.getValue());
		Function1Pt::copy(y_next, this->m_position.getValue());
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

		double newton_first_delta = 0.0;
		double jacobi_first_delta = 0.0;

		for (newton_iteNum = 0; newton_iteNum < newton_maxIterations; ++newton_iteNum) { // newton method loop: H*y_{k+1} = H*y_{k} + gradient of f 

			HM_ComputeFandSdx << <pDims, BLOCK_SIZE >> > (
				m_invK,
				m_F,
				m_Sum_delta_x,
				this->m_position.getValue(),
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				this->weightScale);
			cuSynchronize();

			HM_ComputeFirstPiolaKirchhoff_Linear << <pDims, BLOCK_SIZE >> > (
				m_firstPiolaKirchhoffStress,
				m_F,
				this->m_mu.getValue(),
				this->m_lambda.getValue());
			cuSynchronize();

			HM_ComputeSourceTerm_Linear << <pDims, BLOCK_SIZE >> > (
				m_source_items,
				m_invK,
				m_firstPiolaKirchhoffStress,
				m_position_old,
				y_next,
				m_Sum_delta_x,
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				this->m_mu.getValue(),
				this->m_lambda.getValue(),
				mass, volume, this->getParent()->getDt(),
				this->weightScale);
			cuSynchronize();

			jacobi_convergeFlag = false;
			for (jacobi_iteNum = 0; jacobi_iteNum < jacobi_maxIterations; ++jacobi_iteNum) { // jacobi method loop

				HM_JacobiStep_Linear << <pDims, BLOCK_SIZE >> > (
					y_next,
					y_pre,
					m_source_items,
					m_invK,
					m_Sum_delta_x,
					this->m_restShape.getValue(),
					this->m_horizon.getValue(),
					this->m_mu.getValue(),
					this->m_lambda.getValue(),
					mass, volume,
					this->getParent()->getDt(),
					this->weightScale);
				cuSynchronize();

				{
					Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
					DeviceArray<Real> Delta_y_norm(numOfParticles);
					computeDelta_vec << <pDims, BLOCK_SIZE >> >(y_pre, y_next, Delta_y_norm);
					cuSynchronize();

					Real max_delta = pReduction->maximum(Delta_y_norm.getDataPtr(), numOfParticles);
					Delta_y_norm.release();

					if (jacobi_iteNum == 0) {
						jacobi_first_delta = max_delta;
						if (jacobi_first_delta == 0.0) { jacobi_convergeFlag = true; }
					}
					else {
						if (max_delta / jacobi_first_delta < relative_error_threshold) { jacobi_convergeFlag = true; }
					}
				}

				Function1Pt::copy(y_pre, y_next);
				if (jacobi_convergeFlag) { break; }
			}

			jacobi_total_iteNum += jacobi_iteNum;

			{
				Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
				DeviceArray<Real> Delta_y_norm(numOfParticles);

				computeDelta_vec << <pDims, BLOCK_SIZE >> >(this->m_position.getValue(), y_next, Delta_y_norm);
				cuSynchronize();

				Real max_delta = pReduction->maximum(Delta_y_norm.getDataPtr(), numOfParticles);
				Delta_y_norm.release();

				if (newton_iteNum == 0) {
					newton_first_delta = max_delta;
					if (newton_first_delta == 0.0) { newton_convergeFlag = true; }
				}
				else {
					if (max_delta / newton_first_delta < relative_error_threshold) { newton_convergeFlag = true; }
				}
			}

			HM_UpdatePosition_only << <pDims, BLOCK_SIZE >> > (
				this->m_position.getValue(),
				y_next);
			cuSynchronize();

			if (newton_convergeFlag) { break; }
		}

		HM_UpdatePosition_Velocity << <pDims, BLOCK_SIZE >> > (
			this->m_position.getValue(),
			this->m_velocity.getValue(),
			y_next,
			m_position_old,
			this->getParent()->getDt());
		cuSynchronize();

		y_pre.release();
		y_next.release();

		printf("newton ite num: %d \n jacobi ave_ite num: %lf \n", newton_iteNum + 1 , double(jacobi_total_iteNum + 1) / double(newton_iteNum + 1));
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
		DeviceArray<Coord> y_pre(numOfParticles);
		DeviceArray<Coord> y_next(numOfParticles);

		Function1Pt::copy(y_pre, this->m_position.getValue());
		Function1Pt::copy(y_next, this->m_position.getValue());
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

		double newton_first_delta = 0.0;
		double jacobi_first_delta = 0.0;

		for (newton_iteNum = 0; newton_iteNum < newton_maxIterations; ++newton_iteNum) { // newton method loop: H*y_{k+1} = H*y_{k} + gradient of f 

			HM_ComputeFandSdx << <pDims, BLOCK_SIZE >> > (
				m_invK,
				m_F,
				m_Sum_delta_x,
				this->m_position.getValue(),
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				this->weightScale);
			cuSynchronize();

			HM_ComputeFirstPiolaKirchhoff_StVK << <pDims, BLOCK_SIZE >> > (
				m_firstPiolaKirchhoffStress,
				m_F,
				this->m_mu.getValue(),
				this->m_lambda.getValue());
			cuSynchronize();

			HM_ComputeSourceTerm_StVK << <pDims, BLOCK_SIZE >> > (
				m_source_items,
				m_F,
				m_invK,
				m_firstPiolaKirchhoffStress,
				m_position_old,
				y_next,
				m_Sum_delta_x,
				this->m_restShape.getValue(),
				this->m_horizon.getValue(),
				this->m_mu.getValue(),
				this->m_lambda.getValue(),
				mass, volume, this->getParent()->getDt(),
				this->weightScale);
			cuSynchronize();

			jacobi_convergeFlag = false;
			for (jacobi_iteNum = 0; jacobi_iteNum < jacobi_maxIterations; ++jacobi_iteNum) { // jacobi method loop

				HM_JacobiStep_StVK << <pDims, BLOCK_SIZE >> > (
					y_next,
					y_pre,
					m_source_items,
					m_F,
					m_invK,
					m_Sum_delta_x,
					this->m_restShape.getValue(),
					this->m_horizon.getValue(),
					this->m_mu.getValue(),
					this->m_lambda.getValue(),
					mass, volume,
					this->getParent()->getDt(),
					this->weightScale);
				cuSynchronize();

				{
					Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
					DeviceArray<Real> Delta_y_norm(numOfParticles);
					computeDelta_vec << <pDims, BLOCK_SIZE >> >(y_pre, y_next, Delta_y_norm);
					cuSynchronize();

					Real max_delta = pReduction->maximum(Delta_y_norm.getDataPtr(), numOfParticles);
					Delta_y_norm.release();

					if (jacobi_iteNum == 0) {
						jacobi_first_delta = max_delta;
						if (jacobi_first_delta == 0.0) { jacobi_convergeFlag = true; }
					}
					else {
						if (max_delta / jacobi_first_delta < relative_error_threshold) { jacobi_convergeFlag = true; }
					}
				}

				Function1Pt::copy(y_pre, y_next);
				if (jacobi_convergeFlag) { break; }
			}

			jacobi_total_iteNum += jacobi_iteNum;

			{
				Physika::Reduction<Real>* pReduction = Physika::Reduction<Real>::Create(numOfParticles);
				DeviceArray<Real> Delta_y_norm(numOfParticles);

				computeDelta_vec << <pDims, BLOCK_SIZE >> >(this->m_position.getValue(), y_next, Delta_y_norm);
				cuSynchronize();

				Real max_delta = pReduction->maximum(Delta_y_norm.getDataPtr(), numOfParticles);
				Delta_y_norm.release();

				if (newton_iteNum == 0) {
					newton_first_delta = max_delta;
					if (newton_first_delta == 0.0) { newton_convergeFlag = true; }
				}
				else {
					if (max_delta / newton_first_delta < relative_error_threshold) { newton_convergeFlag = true; }
				}
			}

			HM_UpdatePosition_only << <pDims, BLOCK_SIZE >> > (
				this->m_position.getValue(),
				y_next);
			cuSynchronize();

			if (newton_convergeFlag) { break; }
		}

		HM_UpdatePosition_Velocity << <pDims, BLOCK_SIZE >> > (
			this->m_position.getValue(),
			this->m_velocity.getValue(),
			y_next,
			m_position_old,
			this->getParent()->getDt());
		cuSynchronize();

		y_pre.release();
		y_next.release();

		printf("newton ite num: %d \n jacobi ave_ite num: %lf \n", newton_iteNum + 1, double(jacobi_total_iteNum + 1) / double(newton_iteNum + 1));
		if (jacobi_convergeFlag) { printf("jacobi converge!"); }
		if (newton_convergeFlag) { printf("newton converge!"); }
	}

#ifdef PRECISION_FLOAT
	template class HyperelasticityModule_NewtonMethod<DataType3f>;
#else
	template class HyperelasticityModule_NewtonMethod<DataType3d>;
#endif
}