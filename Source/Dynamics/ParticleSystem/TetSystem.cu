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
		m_tethedrons = std::make_shared<TetrahedronSet<TDataType>>();
		this->setTopologyModule(m_tethedrons);
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

	template<typename TDataType>
	bool TetSystem<TDataType>::initialize()
	{
		return Node::initialize();
	}


	template<typename TDataType>
	void TetSystem<TDataType>::advance(Real dt)
	{
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
	}

	template<typename TDataType>
	void TetSystem<TDataType>::updateTopology()
	{
		
	}


	template<typename TDataType>
	bool TetSystem<TDataType>::resetStatus()
	{
		return Node::resetStatus();
	}


	template<typename TDataType>
	bool TetSystem<TDataType>::translate(Coord t)
	{
		m_tethedrons->translate(t);
		return true;
	}


	template<typename TDataType>
	bool TetSystem<TDataType>::scale(Real s)
	{
		m_tethedrons->scale(s);
		return true;
	}

}