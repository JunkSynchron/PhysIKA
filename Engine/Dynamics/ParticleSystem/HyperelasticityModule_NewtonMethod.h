#pragma once
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
#include "Framework/Topology/NeighborQuery.h"

namespace Physika {

	template<typename TDataType>
	class HyperelasticityModule_NewtonMethod : public ElasticityModule<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef TPair<TDataType> NPair;

		HyperelasticityModule_NewtonMethod();
		~HyperelasticityModule_NewtonMethod() override {};

		enum EnergyType
		{
			Linear,
			StVK
		};

		/**
		* @brief Set the energy function
		*
		*/
		void setEnergyFunction(EnergyType type) { m_energyType = type; }

		void solveElasticity() override;
		void solveElasticity_NewtonMethod();
		void solveElasticity_NewtonMethod_StVK();

		void setInfluenceWeightScale(Real w_scale) { this->weightScale = w_scale; };

	protected:
		bool initializeImpl() override;


		//void previous_enforceElasticity();

	private:
		bool ImplicitMethod = true;

		EnergyType m_energyType;

		DeviceArray<Real> m_totalWeight;
		Real weightScale = 150;

		DeviceArray<Coord> m_Sum_delta_x;
		DeviceArray<Coord> m_source_items;

		NeighborQuery<TDataType> hessian_query;
		DeviceArray<Matrix> m_hessian_matrix;

		DeviceArray<Matrix> m_invK;
		Matrix common_K;

		DeviceArray<Matrix> m_F;
		DeviceArray<Matrix> m_firstPiolaKirchhoffStress;

		bool is_debug = true;
	};

}