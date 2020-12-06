#include <cuda_runtime.h>
#include "TetSystem.h"
#include "Framework/Topology/TetrahedronSet.h"
#include "Core/Utility.h"
#include "Framework/Mapping/FrameToPointSet.h"


namespace PhysIKA
{
	IMPLEMENT_CLASS_1(TetSystem, TDataType)

		template <typename Real, typename Coord>
	__global__ void UpdatePosition(
		DeviceArray<Coord> posArr,
		DeviceArray<Coord> velArr,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		posArr[pId] += dt*velArr[pId];
	}

		template <typename Real, typename Coord>
	__global__ void UpdateVelocity(
		DeviceArray<Coord> velArr,
		DeviceArray<Coord> forceArr,
		DeviceArray<Real> massArr,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;

		velArr[pId] += dt * forceArr[pId] / massArr[pId] + dt * Coord(0.0f, -9.8f, 0.0f);
	}
	
		template <typename Real, typename Coord, typename Matrix>
	__global__ void UpdateAngularVelocity(
		DeviceArray<Coord> angularvelArr,
		DeviceArray<Matrix> invMassArr,
		DeviceArray<Coord> forceMomentArr,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= angularvelArr.size()) return;

		angularvelArr[pId] += dt * (invMassArr[pId]*forceMomentArr[pId]);
	}
		template<typename TDataType>
	TetSystem<TDataType>::TetSystem(std::string name)
		: Node(name)
	{
		//		attachField(&m_velocity, MechanicalState::velocity(), "Storing the particle velocities!", false);
		//		attachField(&m_force, MechanicalState::force(), "Storing the force densities!", false);

		m_tethedrons = std::make_shared<TetrahedronSet<TDataType>>();
		this->setTopologyModule(m_tethedrons);
		Coord trans(0.0, 0.0, 0.0);
		m_tethedrons->setCenter(trans);
		m_tethedrons->setOrientation(m_quaternion.get3x3Matrix());

		// 		m_pointsRender = std::make_shared<PointRenderModule>();
		// 		this->addVisualModule(m_pointsRender);
	}

	template<typename TDataType>
	TetSystem<TDataType>::~TetSystem()
	{

	}


	template<typename TDataType>
	void TetSystem<TDataType>::loadTets(std::string filename)
	{
		m_tethedrons->loadTetFile(filename);
	}
	/*
		template<typename TDataType>
		void TetSystem<TDataType>::loadTets(Coord center, Real r, Real distance)
		{
			std::vector<Coord> vertList;
			std::vector<Coord> normalList;

			Coord lo = center - r;
			Coord hi = center + r;

			for (Real x = lo[0]; x <= hi[0]; x += distance)
			{
				for (Real y = lo[1]; y <= hi[1]; y += distance)
				{
					for (Real z = lo[2]; z <= hi[2]; z += distance)
					{
						Coord p = Coord(x, y, z);
						if ((p - center).norm() < r)
						{
							vertList.push_back(Coord(x, y, z));
						}
					}
				}
			}
			normalList.resize(vertList.size());

			m_pSet->setPoints(vertList);
			m_pSet->setNormals(normalList);

			vertList.clear();
			normalList.clear();
		}

		template<typename TDataType>
		void ParticleSystem<TDataType>::loadParticles(Coord lo, Coord hi, Real distance)
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
		*/
	template<typename TDataType>
	bool TetSystem<TDataType>::translate(Coord t)
	{
		m_tethedrons->translate(t);

		return true;
	}
	template<typename TDataType>
	void TetSystem<TDataType>::setCenter(Coord center)
	{
		currentPosition().setValue(center);
	}

	template<typename TDataType>
	bool TetSystem<TDataType>::scale(Real s)
	{
		m_pSet->scale(s);

		return true;
	}

	template<typename TDataType>
	bool TetSystem<TDataType>::initialize()
	{
		return Node::initialize();
	}


	template<typename TDataType>
	void TetSystem<TDataType>::advance(Real dt)
	{
		//Real mass = currentMass()->getValue();
		//Coord center = currentPosition()->getValue();
		//Coord linearVel = currentVelocity()->getValue();
		//Matrix angularMass = currentAngularMass()->getValue();

		//Coord force = currentForce()->getValue();
		//Coord forceMoment = currentTorque()->getValue();

		//Matrix invMass = angularMass;

		int num = this->currentPosition()->getElementCount();
		cuExecute(num, UpdateVelocity,
			this->currentVelocity()->getValue(),
			this->currentForce()->getValue(),
			this->currentMass()->getValue(),
			dt);

		//currentVelocity()->setValue(linearVel);

		cuExecute(num, UpdatePosition,
			this->currentPosition()->getValue(),
			this->currentVelocity()->getValue(),
			dt);

		//currentPosition()->setValue(center);

		cuExecute(num, UpdateAngularVelocity,
			this->currentAngularVelocity()->getValue(),
			this->currentAngularMass()->getValue(),
			this->currentTorque()->getValue(),
			dt);

		//currentAngularVelocity()->setValue(angularVel);
		Coord angularVel = currentAngularVelocity()->getValue();

		m_quaternion = m_quaternion + (0.5f * dt) * Quaternion<Real>(0, angularVel[0], angularVel[1], angularVel[2])*(m_quaternion);
		m_quaternion.normalize();
		currentOrientation()->setValue(m_quaternion.get3x3Matrix());

	}

	template<typename TDataType>
	void TetSystem<TDataType>::updateTopology()
	{
		
		m_tethedrons->setCenter(currentPosition()->getValue());
		m_tethedrons->setOrientation(m_quaternion.get3x3Matrix());

		auto tMappings = this->getTopologyMappingList();
		for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
		{
			(*iter)->apply();
		}
	}


	template<typename TDataType>
	bool TetSystem<TDataType>::resetStatus()
	{
		auto pts = m_tethedrons->getPoints();

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

	// 	template<typename TDataType>
	// 	std::shared_ptr<PointRenderModule> ParticleSystem<TDataType>::getRenderModule()
	// 	{
	// // 		if (m_pointsRender == nullptr)
	// // 		{
	// // 			m_pointsRender = std::make_shared<PointRenderModule>();
	// // 			this->addVisualModule(m_pointsRender);
	// // 		}
	// 
	// 		return m_pointsRender;
	// 	}
}