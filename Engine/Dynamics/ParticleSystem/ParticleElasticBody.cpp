#include "ParticleElasticBody.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Topology/PointSet.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"
#include "Core/Utility.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/NeighborQuery.h"
#include "ParticleIntegrator.h"
#include "ElasticityModule.h"

#include <random>
#include <ctime>

namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleElasticBody, TDataType)

		template<typename TDataType>
	ParticleElasticBody<TDataType>::ParticleElasticBody(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		m_horizon.setValue(0.02);
		this->attachField(&m_horizon, "horizon", "horizon");

		auto m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		this->getPosition()->connect(m_integrator->m_position);
		this->getVelocity()->connect(m_integrator->m_velocity);
		this->getForce()->connect(m_integrator->m_forceDensity);

		auto m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
		m_horizon.connect(m_nbrQuery->m_radius);
		this->m_position.connect(m_nbrQuery->m_position);

		auto m_elasticity = this->template addConstraintModule<ElasticityModule<TDataType>>("elasticity");
		this->getPosition()->connect(m_elasticity->m_position);
		this->getVelocity()->connect(m_elasticity->m_velocity);
		m_horizon.connect(m_elasticity->m_horizon);
		m_nbrQuery->m_neighborhood.connect(m_elasticity->m_neighborhood);

		//Create a node for surface mesh rendering
		m_surfaceNode = this->template createChild<Node>("Mesh");

		auto triSet = m_surfaceNode->template setTopologyModule<TriangleSet<TDataType>>("surface_mesh");

		m_surfaceRender = m_surfaceNode->template addVisualModule<SurfaceMeshRender>("surface_mesh_render");
		m_surfaceRender->setColor(Vector3f(0.2f, 0.6, 1.0f));
		m_surfaceNode->setVisible(false);

		//Set the topology mapping from PointSet to TriangleSet
		auto surfaceMapping = this->template addTopologyMapping<PointSetToPointSet<TDataType>>("surface_mapping");
		surfaceMapping->setFrom(this->m_pSet);
		surfaceMapping->setTo(triSet);
	}

	template<typename TDataType>
	ParticleElasticBody<TDataType>::~ParticleElasticBody()
	{

	}

	template<typename TDataType>
	bool ParticleElasticBody<TDataType>::translate(Coord t)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}

	template<typename TDataType>
	bool ParticleElasticBody<TDataType>::scale(Real s)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}


	template<typename TDataType>
	bool ParticleElasticBody<TDataType>::initialize()
	{
		return ParticleSystem<TDataType>::initialize();
	}

	template<typename TDataType>
	void ParticleElasticBody<TDataType>::advance(Real dt)
	{
		auto integrator = this->template getModule<ParticleIntegrator<TDataType>>("integrator");

		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");

		integrator->begin();

		integrator->integrate();

		if (module != nullptr)
			module->constrain();

		integrator->end();
	}

	template<typename TDataType>
	void ParticleElasticBody<TDataType>::updateTopology()
	{
		auto pts = this->m_pSet->getPoints();
		Function1Pt::copy(pts, this->getPosition()->getValue());

		auto tMappings = this->getTopologyMappingList();
		for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
		{
			(*iter)->apply();
		}
	}


	template<typename TDataType>
	std::shared_ptr<ElasticityModule<TDataType>> ParticleElasticBody<TDataType>::getElasticitySolver()
	{
		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");
		return module;
	}


	template<typename TDataType>
	void ParticleElasticBody<TDataType>::setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver)
	{
		auto nbrQuery = this->template getModule<NeighborQuery<TDataType>>("neighborhood");
		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");

		this->getPosition()->connect(solver->m_position);
		this->getVelocity()->connect(solver->m_velocity);
		nbrQuery->m_neighborhood.connect(solver->m_neighborhood);
		m_horizon.connect(solver->m_horizon);

		this->deleteModule(module);

		solver->setName("elasticity");
		this->addConstraintModule(solver);
	}


	template<typename TDataType>
	void ParticleElasticBody<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
	}


	template<typename TDataType>
	std::shared_ptr<PointSetToPointSet<TDataType>> ParticleElasticBody<TDataType>::getTopologyMapping()
	{
		auto mapping = this->template getModule<PointSetToPointSet<TDataType>>("surface_mapping");

		return mapping;
	}

	template<typename TDataType>
	void ParticleElasticBody<TDataType>::loadParticles_randomOffset(Coord lo, Coord hi, Real distance, Real offset_rate) {
		std::vector<Coord> vertList;
		std::vector<Coord> normalList;

		std::default_random_engine random_e(time(0));
		std::uniform_real_distribution<double> u(-1.0, 1.0);

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p = Coord(x + u(random_e)*offset_rate*distance, y, z);
					vertList.push_back(p);
				}
			}
		}
		normalList.resize(vertList.size());

		m_pSet->setPoints(vertList);
		m_pSet->setNormals(normalList);

		std::cout << "particle number: " << vertList.size() << std::endl;

		vertList.clear();
		normalList.clear();
	}

}