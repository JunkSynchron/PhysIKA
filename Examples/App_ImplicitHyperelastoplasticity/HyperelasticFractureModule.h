#pragma once
#include "HyperelastoplasticityModule.h"

namespace PhysIKA {

	template<typename TDataType> class SummationDensity;

	template<typename TDataType>
	class HyperelasticFractureModule : public HyperelastoplasticityModule<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef TPair<TDataType> NPair;

		HyperelasticFractureModule();
		~HyperelasticFractureModule() override {};

		void applyPlasticity() override;
	};

#ifdef PRECISION_FLOAT
	template class HyperelasticFractureModule<DataType3f>;
#else
	template class HyperelasticFractureModule<DataType3d>;
#endif
}