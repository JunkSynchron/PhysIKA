/**
 * @file HyperelasticityModule.h
 * @author Xiaowei He (xiaowei@iscas.ac.cn)
 * @brief This is an implementation of hyperelasticity based on a set of basis functions.
 * 		  For more details, please refer to [Xu et al. 2018] "Reformulating Hyperelastic Materials with Peridynamic Modeling"
 * @version 0.1
 * @date 2019-06-18
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once
#include "ElasticityModule.h"
#include "Core/Utility/Reduction.h"

namespace Physika {

	/**
	 * @brief Basis Function
	 * 
	 * @tparam T Real value
	 * @tparam n The degree of the basis
	 */

	/***********************************************************************************************
	template<typename T, int n>
	class Basis
	{
	public:
		static COMM_FUNC T A(T s) {
			return ((pow(s, n + 1) - 1) / (n + 1) + (pow(s, 1 - n) - 1) / (n - 1)) / n;
		}
		
		static COMM_FUNC T B(T s) {
			return 2 * ((pow(s, n + 1) - 1) / (n + 1) - s + 1) / n;
		}

		static COMM_FUNC T dA(T s) {
			Real sn = pow(s, n);
			return (sn - 1 / sn) / 2;
		}

		static COMM_FUNC T dB(T s) {
			return 2 * (pow(s, n) - 1);
		}
	};

	template<typename T>
	class Basis<T, 1>
	{
	public:
		static COMM_FUNC T A(T s) {
			return (s * s - 1) / 2 - log(s);
		}

		static	COMM_FUNC T B(T s) {
			return s * s - s;
		}

		static COMM_FUNC T dA(T s) {
			return s - 1 / s;
		}

		static COMM_FUNC T dB(T s) {
			return 2 * (s - 1);
		}
	};
	**************************************************************************/

	enum EnergyType
	{
		Linear,
		Quadratic
	};

	template<typename Real, typename Matrix>
	class HyperelasticityModel
	{
	public:
		COMM_FUNC HyperelasticityModel() {};

		COMM_FUNC virtual Real getEnergy(Real lambda1, Real lambda2, Real lambda3) = 0;
		COMM_FUNC virtual Matrix getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3) = 0;
		COMM_FUNC virtual Matrix getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3) = 0;

		Real density;
	};

	template<typename Real, typename Matrix>
	class StVKModel : public HyperelasticityModel<Real, Matrix>
	{
	public:
		COMM_FUNC StVKModel() : HyperelasticityModel<Real, Matrix>()
		{
			density = Real(1);
			mu = Real(48000);
			lambda = Real(12000);
		}

		COMM_FUNC virtual Real getEnergy(Real lambda1, Real lambda2, Real lambda3) override
		{
			Real I = lambda1*lambda1 + lambda2*lambda2 + lambda3*lambda3;
			Real sq1 = lambda1*lambda1;
			Real sq2 = lambda2*lambda2;
			Real sq3 = lambda3*lambda3;
			Real II = sq1*sq1 + sq2*sq2 + sq3*sq3;
			return 0.5*lambda*(I - 3)*(I - 3) + 0.25*mu*(II - 2 * I + 3);
		}

		COMM_FUNC virtual Matrix getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3) override
		{
			Real I = lambda1*lambda1 + lambda2*lambda2 + lambda3*lambda3;

			Real D1 = 2 * lambda*I + mu*lambda1*lambda1;
			Real D2 = 2 * lambda*I + mu*lambda2*lambda2;
			Real D3 = 2 * lambda*I + mu*lambda3*lambda3;

			Matrix D;
			D(0, 0) = D1;
			D(1, 1) = D2;
			D(2, 2) = D3;
			return D;
		}

		COMM_FUNC virtual Matrix getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3) override
		{
			Matrix D = (6 * lambda + mu)*Matrix::identityMatrix();
			return D;
		}

		Real lambda;
		Real mu;
	};



	template<typename TDataType>
	class HyperelasticityModule_test : public ElasticityModule<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef TPair<TDataType> NPair;

		HyperelasticityModule_test();
		~HyperelasticityModule_test() override {};

		/**
		 * @brief Set the energy function
		 * 
		 */
		void setEnergyFunction(EnergyType type) { m_energyType = type; }

		void solveElasticity() override;

		void solveElasticityImplicit();

		void solveElasticityGradientDescent();

	protected:
		bool initializeImpl() override;

		void initializeVolume();

		//void previous_enforceElasticity();

	private:
		void getEnergy(Real& totalEnergy, DeviceArray<Coord>& position);

		EnergyType m_energyType;

		DeviceArray<Real> m_energy;
		DeviceArray<Real> m_volume;
		DeviceArray<Coord> m_gradient;

		DeviceArray<Coord> m_eigenValues;

		DeviceArray<Matrix> m_F;
		DeviceArray<Matrix> m_invF;
		DeviceArray<Matrix> m_Rot;

		DeviceArray<Coord> y_pre;
		DeviceArray<Coord> y_next;

		Reduction<Real>* m_reduce;
	};

}