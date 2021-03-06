#include "HyperelasticBody.h"
#include "Framework/Topology/TetrahedronSet.h"
#include "Framework/Topology/UnstructuredPointSet.h"
#include "Core/Utility.h"
#include "Framework/Topology/NeighborQuery.h"
#include "ParticleIntegrator.h"
#include "HyperelasticityModule_test.h"

#include "IO/Smesh_IO/smesh.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(HyperelasticBody, TDataType)

	template<typename TDataType>
	HyperelasticBody<TDataType>::HyperelasticBody(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		m_pSet = std::make_shared<UnstructuredPointSet<TDataType>>();
		this->setTopologyModule(m_pSet);

		this->varHorizon()->setValue(0.0085);
		//		this->attachField(&m_horizon, "horizon", "horizon");

		m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		this->currentPosition()->connect(m_integrator->inPosition());
		this->currentVelocity()->connect(m_integrator->inVelocity());
		this->currentForce()->connect(m_integrator->inForceDensity());

		this->getAnimationPipeline()->push_back(m_integrator);

		m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
		this->varHorizon()->connect(m_nbrQuery->inRadius());
		this->currentPosition()->connect(m_nbrQuery->inPosition());

		this->getAnimationPipeline()->push_back(m_nbrQuery);


		m_hyper = this->template addConstraintModule<HyperelasticityModule_test<TDataType>>("elasticity");
		this->varHorizon()->connect(m_hyper->inHorizon());
		this->currentPosition()->connect(m_hyper->inPosition());
		this->currentVelocity()->connect(m_hyper->inVelocity());
		m_nbrQuery->outNeighborhood()->connect(m_hyper->inNeighborhood());

		this->getAnimationPipeline()->push_back(m_hyper);

		//Create a node for surface mesh rendering
		m_mesh_node = std::make_shared<TetSystem<TDataType>>("Mesh");
		this->addChild(m_mesh_node);

		//Set the topology mapping from PointSet to TriangleSet
	}

	template<typename TDataType>
	HyperelasticBody<TDataType>::~HyperelasticBody()
	{
		
	}

	template<typename TDataType>
	bool HyperelasticBody<TDataType>::translate(Coord t)
	{
		m_pSet->translate(t);

		this->getMeshNode()->translate(t);

		return true;
	}

	template<typename TDataType>
	bool HyperelasticBody<TDataType>::scale(Real s)
	{
		m_pSet->scale(s);

		this->getMeshNode()->scale(s);

		return true;
	}


	template<typename TDataType>
	bool HyperelasticBody<TDataType>::initialize()
	{
		return true;
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::advance(Real dt)
	{
		m_integrator->begin();

		m_integrator->integrate();

		m_hyper->constrain();
		

		m_integrator->end();
	}

	//夏提完成，根据UnstructuredPointSet中m_coords和m_pointNeighbors来更新TetrahedronSet
	template<typename TDataType>
	void HyperelasticBody<TDataType>::updateTopology()
	{
		if (!this->currentPosition()->isEmpty())
		{
			int num = this->currentPosition()->getElementCount();
			auto& pts = m_pSet->getPoints();
			if (num != pts.size())
			{
				pts.resize(num);
			}

			Function1Pt::copy(pts, this->currentPosition()->getValue());
		}
	}

	template<typename TDataType>
	bool PhysIKA::HyperelasticBody<TDataType>::resetStatus()
	{
		auto pts = m_pSet->getPoints();

		if (pts.size() > 0)
		{
			this->currentPosition()->setElementCount(pts.size());
			this->currentVelocity()->setElementCount(pts.size());
			this->currentForce()->setElementCount(pts.size());

			Function1Pt::copy(this->currentPosition()->getValue(), pts);
			this->currentVelocity()->getReference()->reset();
		}

		return Node::resetStatus();
	}

	template<typename TDataType>
	std::shared_ptr<ElasticityModule<TDataType>> HyperelasticBody<TDataType>::getElasticitySolver()
	{
		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");
		return module;
	}


	template<typename TDataType>
	void HyperelasticBody<TDataType>::setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver)
	{
		auto nbrQuery = this->template getModule<NeighborQuery<TDataType>>("neighborhood");
		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");

		this->currentPosition()->connect(solver->inPosition());
		this->currentVelocity()->connect(solver->inVelocity());
		nbrQuery->outNeighborhood()->connect(solver->inNeighborhood());
		this->varHorizon()->connect(solver->inHorizon());

		this->deleteModule(module);
		
		solver->setName("elasticity");
		this->addConstraintModule(solver);
	}


	template<typename TDataType>
	void HyperelasticBody<TDataType>::loadFile(std::string filename)
	{
		Smesh meshLoader;
		meshLoader.loadFile(filename);

		this->getMeshNode()->derivedTopology()->setPoints(meshLoader.m_points);
		this->getMeshNode()->derivedTopology()->setTetrahedrons(meshLoader.m_tets);

		std::vector<Vector3f> centroids;
		centroids.resize(meshLoader.m_tets.size());

		for (int i= 0; i < meshLoader.m_tets.size(); i++)
		{
			auto tet = meshLoader.m_tets[i];
			centroids[i] = (meshLoader.m_points[tet[0]] + meshLoader.m_points[tet[1]] + meshLoader.m_points[tet[2]] + meshLoader.m_points[tet[3]]) / 4;
		}

		m_pSet->setPoints(centroids);
	}



	template<typename TDataType>
	void HyperelasticBody<TDataType>::loadParticles(Coord lo, Coord hi, Real distance)
	{
		std::vector<Coord> vertList;
		std::vector<Coord> normalList;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p = Coord(x, y, z);
					vertList.push_back(Coord(x, y, z));
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