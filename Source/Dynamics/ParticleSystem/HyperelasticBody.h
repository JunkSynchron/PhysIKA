#pragma once
#include "Dynamics/ParticleSystem/ParticleSystem.h"
#include "TetSystem.h"

namespace PhysIKA
{
	template<typename> class ElasticityModule;
	template<typename> class ParticleIntegrator;
	template<typename> class UnstructuredPointSet;
	template<typename> class NeighborQuery;
	template<typename> class HyperelasticityModule_test;

	/*!
	*	\class	ParticleElasticBody
	*	\brief	Peridynamics-based elastic object.
	*/
	template<typename TDataType>
	class HyperelasticBody : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleElasticBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		HyperelasticBody(std::string name = "default");
		virtual ~HyperelasticBody();

		bool initialize() override;
		void advance(Real dt) override;
		void updateTopology() override;
		bool resetStatus() override;

		virtual bool translate(Coord t);
		virtual bool scale(Real s);

		void setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver);

		std::shared_ptr<ElasticityModule<TDataType>> getElasticitySolver();

		void loadFile(std::string filename);

		std::shared_ptr<TetSystem<TDataType>> getMeshNode() { return m_mesh_node; }

		void loadParticles(Coord lo, Coord hi, Real distance);

	public:
		DEF_EMPTY_VAR(Horizon, Real, "Horizon");

	protected:
		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<NeighborQuery<TDataType>> m_nbrQuery;
		std::shared_ptr<HyperelasticityModule_test<TDataType>> m_hyper;

		std::shared_ptr<UnstructuredPointSet<TDataType>> m_pSet;

		std::shared_ptr<TetSystem<TDataType>> m_mesh_node;
	};


#ifdef PRECISION_FLOAT
	template class HyperelasticBody<DataType3f>;
#else
	template class HyperelasticBody<DataType3d>;
#endif
}