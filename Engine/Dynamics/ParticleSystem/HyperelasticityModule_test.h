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
		
		enum EnergyType
		{
			Linear,
			Quadratic
		};

		/**
		 * @brief Set the energy function
		 * 
		 */
		void setEnergyFunction(EnergyType type) { m_energyType = type; }

		void solveElasticity() override;

		void solveElasticityExplicit();
		void solveElasticityImplicit();

	protected:
		bool initializeImpl() override;

		//void previous_enforceElasticity();

	private:
		EnergyType m_energyType;

		DeviceArray<Coord> m_position_old;

		DeviceArray<Matrix> m_invK;
		DeviceArray<Matrix> m_invL;

		DeviceArray<Matrix> m_F;
		DeviceArray<Matrix> m_invF;
		DeviceArray<Matrix> m_firstPiolaKirchhoffStress;
	};

}