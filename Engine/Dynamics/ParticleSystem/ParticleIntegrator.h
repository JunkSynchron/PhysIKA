#pragma once
#include "Framework/Framework/NumericalIntegrator.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Vector.h"

namespace Physika {
	template<typename TDataType>
	class ParticleIntegrator : public NumericalIntegrator
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleIntegrator();
		~ParticleIntegrator() override {};
		
		void begin() override;
		void end() override;

		bool integrate() override;

		bool updateVelocity();
		bool updatePosition();

		void disableGravity() { exit_gravity = false; }
		void enableGravity() { exit_gravity = true; }
		void setGravity(Coord g) { gravity = g; }
		void setFixedStretchForce(Real f) { extra_force = f; }
		void setFixedStretchOffset(Real v) { extra_stretch = v; }
		void setBarSize(Vector<int, 3> xyz) { bar_xyz = xyz; }

	protected:
		bool initializeImpl() override;

	public:
		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Coord> m_forceDensity;

		Vector<int, 3> bar_xyz;

		bool exit_gravity = true;
		Real extra_force = Real(0.0);
		Real extra_stretch = Real(0.0);
		Coord gravity = Coord(0.0, -9.8, 0.0);

	private:
		DeviceArray<Coord> m_prePosition;
		DeviceArray<Coord> m_preVelocity;

	};

#ifdef PRECISION_FLOAT
	template class ParticleIntegrator<DataType3f>;
#else
 	template class ParticleIntegrator<DataType3d>;
#endif
}