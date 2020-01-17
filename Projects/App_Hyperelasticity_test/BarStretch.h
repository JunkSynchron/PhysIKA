#pragma once
#include "Dynamics/ParticleSystem/ParticleIntegrator.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Vector.h"

namespace Physika {
	template<typename TDataType>
	class BarStretchIntegrator : public ParticleIntegrator<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		BarStretchIntegrator();
		~BarStretchIntegrator() {};

		bool integrate() override;

		bool updateVelocity() ;
		bool updatePosition() ;

	protected:
		bool initializeImpl() override;

	public:
		void disableGravity() { exit_gravity = false; }
		void enableGravity() { exit_gravity = true; }
		void setGravity(Coord g) { gravity = g; }
		void setFixedStretchForce(Real f) { extra_force = f; }
		void setFixedStretchOffset(Real v) { extra_stretch = v; }
		void setBarSize(Vector<int, 3> xyz) { bar_xyz = xyz; }
		void setInitialStretch(Real rate) { initial_stretch_rate = rate; }
		void setRelative_YPlane(Real y) { relative_y = y; }

		Vector<int, 3> bar_xyz;

		bool first_call = true;
		bool exit_gravity = true;
		Real initial_stretch_rate = 1.0;
		Real relative_y = 0.5;
		Real extra_force = Real(0.0);
		Real extra_stretch = Real(0.0);
		Coord gravity = Coord(0.0, -9.8, 0.0);
	};

#ifdef PRECISION_FLOAT
	template class BarStretchIntegrator<DataType3f>;
#else
	template class BarStretchIntegrator<DataType3d>;
#endif
}