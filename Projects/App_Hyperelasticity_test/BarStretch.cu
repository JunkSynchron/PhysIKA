#include <cuda_runtime.h>
#include "BarStretch.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/Node.h"
#include "Core/Utility.h"
#include "Framework/Framework/SceneGraph.h"

namespace Physika {
	template<typename TDataType>
	BarStretchIntegrator<TDataType>::BarStretchIntegrator()
		: ParticleIntegrator()
	{
		
	}

	template<typename TDataType>
	bool BarStretchIntegrator<TDataType>::initializeImpl()
	{
		return ParticleIntegrator::initializeImpl();
	}

	template<typename Real, typename Coord>
	__global__ void K_SetInitialStretch_y(
		DeviceArray<Coord> pos,
		Vector<int, 3> bar_xyz,
		Real relative_y,
		Real stretch_rate)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		Coord initial_pos = pos[pId];
		Real stretch_delta_y = stretch_rate * (initial_pos[1] - relative_y);
		initial_pos[1] = relative_y + stretch_delta_y;
		pos[pId] = initial_pos;
	}

	template<typename TDataType>
	bool BarStretchIntegrator<TDataType>::integrate()
	{
		if (first_call) {
			first_call = false;
			printf("Bar stretch first call\n");
			cuint pDims = cudaGridSize(m_position.getReference()->size(), BLOCK_SIZE);

			if (this->initial_stretch_rate != 1.0) {
				K_SetInitialStretch_y << <pDims, BLOCK_SIZE >> > (
					m_position.getValue(),
					this->bar_xyz,
					this->relative_y,
					this->initial_stretch_rate);
				cuSynchronize();
			}
		}

		updateVelocity();
		updatePosition();

		return true;
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DeviceArray<Coord> vel,
		DeviceArray<Coord> forceDensity,
		Real gravity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forceDensity.size()) return;

		vel[pId] += dt * (forceDensity[pId] + Coord(0, gravity, 0));
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DeviceArray<Coord> vel,
		DeviceArray<Coord> force,
		DeviceArray<Real> mass,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= force.size()) return;

		vel[pId] += dt * force[pId] / mass[pId];
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity_fixed(
		DeviceArray<Coord> vel,
		DeviceArray<Coord> forceDensity,
		Real gravity,
		Real dt,
		Vector<int, 3> bar_xyz)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forceDensity.size()) return;

		vel[pId] += dt * (forceDensity[pId] + Coord(0, gravity, 0));

		int total_size = bar_xyz[0] * bar_xyz[1] * bar_xyz[2];
		int yz_cross_section_size = bar_xyz[1] * bar_xyz[2];
		int yz_plane_index = pId % yz_cross_section_size;
		int z_size = bar_xyz[2];

		if (yz_plane_index + z_size >= yz_cross_section_size) {
			vel[pId] = Coord(0.0); // keep still
		}
		else {
			// do nothing
		}
	}

	template<typename TDataType>
	bool BarStretchIntegrator<TDataType>::updateVelocity()
	{
		Real dt = getParent()->getDt();
		Real gravity = SceneGraph::getInstance().getGravity();
		cuint pDims = cudaGridSize(m_position.getReference()->size(), BLOCK_SIZE);

		if (!exit_gravity) { gravity = Real(0.0); }

		if (initial_stretch_rate != 0.0) {
			K_UpdateVelocity_fixed << <pDims, BLOCK_SIZE >> > (
				m_velocity.getValue(),
				m_forceDensity.getValue(),
				gravity,
				dt,
				bar_xyz);
			cuSynchronize()
		}
		else {
			K_UpdateVelocity << <pDims, BLOCK_SIZE >> > (
				m_velocity.getValue(),
				m_forceDensity.getValue(),
				gravity,
				dt);
			cuSynchronize();
		}

		return true;
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DeviceArray<Coord> pos,
		DeviceArray<Coord> vel,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		pos[pId] += dt * vel[pId];
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdatePosition_fixed(
		DeviceArray<Coord> pos,
		DeviceArray<Coord> vel,
		Real dt,
		Vector<int, 3> bar_xyz)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		int total_size = bar_xyz[0] * bar_xyz[1] * bar_xyz[2];
		int yz_cross_section_size = bar_xyz[1] * bar_xyz[2];
		int yz_plane_index = pId % yz_cross_section_size;
		int z_size = bar_xyz[2];

		if (yz_plane_index + z_size >= yz_cross_section_size) {
			// do nothing; keep still
		}
		else {
			pos[pId] += dt * vel[pId];
		}
	}

	template<typename TDataType>
	bool BarStretchIntegrator<TDataType>::updatePosition()
	{
		Real dt = getParent()->getDt();
		cuint pDims = cudaGridSize(m_position.getReference()->size(), BLOCK_SIZE);

		if (this->initial_stretch_rate != 1.0) {
			K_UpdatePosition_fixed << <pDims, BLOCK_SIZE >> > (
				m_position.getValue(),
				m_velocity.getValue(),
				dt,
				bar_xyz);
			cuSynchronize();
		}
		else {
			K_UpdatePosition << <pDims, BLOCK_SIZE >> > (
				m_position.getValue(),
				m_velocity.getValue(),
				dt);
			cuSynchronize();
		}

		return true;
	}
}