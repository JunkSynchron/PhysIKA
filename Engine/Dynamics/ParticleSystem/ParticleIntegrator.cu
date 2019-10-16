#include <cuda_runtime.h>
#include "ParticleIntegrator.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/Node.h"
#include "Core/Utility.h"
#include "Framework/Framework/SceneGraph.h"

namespace Physika
{
	template<typename TDataType>
	ParticleIntegrator<TDataType>::ParticleIntegrator()
		: NumericalIntegrator()
	{
		attachField(&m_position, "position", "Storing the particle positions!", false);
		attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		attachField(&m_forceDensity, "force", "Particle forces", false);
	}

	template<typename TDataType>
	void ParticleIntegrator<TDataType>::begin()
	{
		Function1Pt::copy(m_prePosition, m_position.getValue());
		Function1Pt::copy(m_preVelocity, m_velocity.getValue());
		
		m_forceDensity.getReference()->reset();
	}


	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::initializeImpl()
	{
		if (!isAllFieldsReady())
		{
			std::cout << "Exception: " << std::string("DensitySummation's fields are not fully initialized!") << "\n";
			return false;
		}

		int num = m_position.getElementCount();

		m_prePosition.resize(num);
		m_preVelocity.resize(num);


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
	__global__ void K_UpdateVelocity_extra_fixed_force(
		DeviceArray<Coord> vel,
		DeviceArray<Coord> forceDensity,
		Real gravity,
		Real extra_force_density,
		Real dt,
		Vector<int,3> bar_xyz)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forceDensity.size()) return;

		vel[pId] += dt * (forceDensity[pId] + Coord(0, gravity, 0));

		int total_size = bar_xyz[0] * bar_xyz[1] * bar_xyz[2];
		int cross_section_size = bar_xyz[1] * bar_xyz[2];

		if (pId < cross_section_size) {
			
			vel[pId] += dt * Coord(-extra_force_density, 0, 0);
		}
		if (total_size - pId <= cross_section_size) {
			
			vel[pId] += dt * Coord(extra_force_density, 0, 0);
		}
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

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::updateVelocity()
	{
		Real dt = getParent()->getDt();
		Real gravity = SceneGraph::getInstance().getGravity();
		cuint pDims = cudaGridSize(m_position.getReference()->size(), BLOCK_SIZE);

		if (!exit_gravity) { gravity = Real(0.0); }

		if (extra_force > 0.0) {
			K_UpdateVelocity_extra_fixed_force << <pDims, BLOCK_SIZE >> > (
				m_velocity.getValue(),
				m_forceDensity.getValue(),
				gravity,
				extra_force,
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
	__global__ void K_UpdatePosition_fixed_offset(
		DeviceArray<Coord> pos,
		DeviceArray<Coord> vel,
		DeviceArray<Coord> pre_pos,
		Real stretch_velocity,
		Real dt,
		bool end_phase_call,
		Vector<int, 3> bar_xyz)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		int total_size = bar_xyz[0] * bar_xyz[1] * bar_xyz[2];
		int cross_section_size = bar_xyz[1] * bar_xyz[2];

		if (pId < cross_section_size) {
			
			if (end_phase_call) {
				pos[pId] = pre_pos[pId] + dt * Coord(-stretch_velocity, 0, 0);
			}
		}
		else if (total_size - pId <= cross_section_size) {
			if (end_phase_call) {
				pos[pId] = pre_pos[pId] + dt * Coord(stretch_velocity, 0, 0);
			}
		}
		else {
			if (!end_phase_call) {
				pos[pId] += dt * vel[pId];
			}
		}
	}

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::updatePosition()
	{
		Real dt = getParent()->getDt();
		cuint pDims = cudaGridSize(m_position.getReference()->size(), BLOCK_SIZE);

		if (extra_stretch > 0.0) {
			K_UpdatePosition_fixed_offset << <pDims, BLOCK_SIZE >> > (
				m_position.getValue(),
				m_velocity.getValue(),
				m_prePosition,
				extra_stretch,
				dt,
				false,
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

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::integrate()
	{
		updateVelocity();
		updatePosition();

		return true;
	}

	template<typename TDataType>
	void ParticleIntegrator<TDataType>::end()
	{
		Real dt = getParent()->getDt();
		cuint pDims = cudaGridSize(m_position.getReference()->size(), BLOCK_SIZE);

		if (extra_stretch > 0.0) {
			K_UpdatePosition_fixed_offset << <pDims, BLOCK_SIZE >> > (
				m_position.getValue(),
				m_velocity.getValue(),
				m_prePosition,
				extra_stretch,
				dt,
				true,
				bar_xyz);
			cuSynchronize();
		}
	}

}