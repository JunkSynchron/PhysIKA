#include "HyperelastoplasticityBody.h"
#include "Dynamics/ParticleSystem/PositionBasedFluidModel.h"

#include "Framework/Topology/TriangleSet.h"
#include "Framework/Topology/UnstructuredPointSet.h"

#include "Core/Utility.h"
#include "Dynamics/ParticleSystem/Peridynamics.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/NeighborQuery.h"
#include "Dynamics/ParticleSystem/ParticleIntegrator.h"
#include "Dynamics/ParticleSystem/ElastoplasticityModule.h"

#include "Dynamics/ParticleSystem/DensityPBD.h"
#include "Dynamics/ParticleSystem/ImplicitViscosity.h"

#include "HyperelasticFractureModule.h"


namespace PhysIKA
{
	IMPLEMENT_CLASS_1(HyperelastoplasticityBody, TDataType)

	template<typename TDataType>
	HyperelastoplasticityBody<TDataType>::HyperelastoplasticityBody(std::string name)
		: HyperelasticBody<TDataType>(name)
	{
		m_plasticity = this->template addConstraintModule<HyperelastoplasticityModule<TDataType>>("elastoplasticity");
		this->currentPosition()->connect(m_plasticity->inPosition());
		this->currentVelocity()->connect(m_plasticity->inVelocity());
		m_nbrQuery->outNeighborhood()->connect(m_plasticity->inNeighborhood());

		m_fracture = this->template addConstraintModule<HyperelasticFractureModule<TDataType>>("fracture");
		this->currentPosition()->connect(m_fracture->inPosition());
		this->currentVelocity()->connect(m_fracture->inVelocity());
		m_nbrQuery->outNeighborhood()->connect(m_fracture->inNeighborhood());

// 		m_elasticity = this->template addConstraintModule<ElasticityModule<TDataType>>("elasticity");
// 		this->currentPosition()->connect(m_elasticity->inPosition());
// 		this->currentVelocity()->connect(m_elasticity->inVelocity());
// 		m_nbrQuery->outNeighborhood()->connect(m_elasticity->inNeighborhood());

		m_pbdModule = this->template addConstraintModule<DensityPBD<TDataType>>("pbd");
		this->varHorizon()->connect(m_pbdModule->varSmoothingLength());
		this->currentPosition()->connect(m_pbdModule->inPosition());
		this->currentVelocity()->connect(m_pbdModule->inVelocity());
		m_nbrQuery->outNeighborhood()->connect(m_pbdModule->inNeighborIndex());

		m_visModule = this->template addConstraintModule<ImplicitViscosity<TDataType>>("viscosity");
		m_visModule->setViscosity(Real(1));
		this->varHorizon()->connect(&m_visModule->m_smoothingLength);
		this->currentPosition()->connect(&m_visModule->m_position);
		this->currentVelocity()->connect(&m_visModule->m_velocity);
		m_nbrQuery->outNeighborhood()->connect(&m_visModule->m_neighborhood);


		m_surfaceNode = this->template createChild<Node>("Mesh");
		m_surfaceNode->setVisible(false);

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		m_surfaceNode->setTopologyModule(triSet);

// 		std::shared_ptr<PointSetToPointSet<TDataType>> surfaceMapping = std::make_shared<PointSetToPointSet<TDataType>>(this->m_pSet, triSet);
// 		this->addTopologyMapping(surfaceMapping);
	}

	template<typename TDataType>
	HyperelastoplasticityBody<TDataType>::~HyperelastoplasticityBody()
	{
		
	}

	template<typename TDataType>
	void HyperelastoplasticityBody<TDataType>::advance(Real dt)
	{
		auto module = this->template getModule<HyperelastoplasticityModule<TDataType>>("elastoplasticity");

		m_integrator->begin();

		m_integrator->integrate();

		m_nbrQuery->compute();
		module->solveElasticity();
		//m_elasticity->solveElasticity();

		m_nbrQuery->compute();

		module->applyPlasticity();

		m_visModule->constrain();

		m_integrator->end();
	}

	template<typename TDataType>
	void HyperelastoplasticityBody<TDataType>::updateTopology()
	{
		auto pts = this->m_pSet->getPoints();
		Function1Pt::copy(pts, this->currentPosition()->getValue());

		auto tMappings = this->getTopologyMappingList();
		for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
		{
			(*iter)->apply();
		}
	}

	template<typename TDataType>
	bool HyperelastoplasticityBody<TDataType>::initialize()
	{
		m_nbrQuery->initialize();
		m_nbrQuery->compute();

		return HyperelasticBody<TDataType>::initialize();
	}

	template<typename TDataType>
	void HyperelastoplasticityBody<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
	}

	template<typename TDataType>
	bool HyperelastoplasticityBody<TDataType>::translate(Coord t)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return HyperelasticBody<TDataType>::translate(t);
	}

	template<typename TDataType>
	bool HyperelastoplasticityBody<TDataType>::scale(Real s)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return HyperelasticBody<TDataType>::scale(s);
	}


/*	template<typename TDataType>
	void HyperelastoplasticityBody<TDataType>::setElastoplasticitySolver(std::shared_ptr<ElastoplasticityModule<TDataType>> solver)
	{
		auto module = this->getModule("elastoplasticity");
		this->deleteModule(module);

		auto nbrQuery = this->template getModule<NeighborQuery<TDataType>>("neighborhood");

		this->currentPosition()->connect(solver->inPosition());
		this->currentVelocity()->connect(solver->inVelocity());
		nbrQuery->outNeighborhood()->connect(solver->inNeighborhood());
		m_horizon.connect(solver->inHorizon());

		solver->setName("elastoplasticity");
		this->addConstraintModule(solver);
	}*/
}