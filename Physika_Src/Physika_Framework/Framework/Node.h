#pragma once
#include "Base.h"
#include "Physika_Core/Typedef.h"
#include "FieldVar.h"
#include "Physika_Core/Platform.h"
#include "NumericalModel.h"
#include "ModuleTopology.h"
#include "DeviceContext.h"
#include "MechanicalState.h"
#include "ModuleForce.h"
#include "ModuleConstraint.h"
#include "CollisionModel.h"
#include "CollidableObject.h"
#include "ModuleVisual.h"
#include "ControllerAnimation.h"
#include "ControllerRender.h"
#include "TopologyMapping.h"

namespace Physika {
class Action;

class Node : public Base
{
	DECLARE_CLASS(Node)
public:

	template<class T>
	using SPtr = std::shared_ptr<T>;

	Node(std::string name = "default");
	virtual ~Node();

	void setName(std::string name);
	std::string getName();

	Node* getChild(std::string name);
	Node* getParent();
	Node* getRoot();

	/// Check the state of dynamics
	virtual bool isActive();

	/// Set the state of dynamics
	virtual void setActive(bool active);

	/// Check the visibility of context
	virtual bool isVisible();

	/// Set the visibility of context
	virtual void setVisible(bool visible);

	/// Simulation time
	virtual Real getTime();

	/// Simulation timestep
	virtual Real getDt();

	void setDt(Real dt);

	void setGravity(Real g);
	Real getGravity();

	void setMass(Real mass);
	Real getMass();

	template<class TNode>
	std::shared_ptr<TNode> createChild(std::string name)
	{
		return addChild(TypeInfo::New<TNode>(name));
	}

	std::shared_ptr<Node> addChild(std::shared_ptr<Node> child) {
		m_children.push_back(child);
		child->setParent(this);
		return child;
	}

	void removeChild(std::shared_ptr<Node> child);

	ListPtr<Node> getChildren() { return m_children; }

	std::shared_ptr<DeviceContext> getContext();
	void setContext(std::shared_ptr<DeviceContext> context);

	std::shared_ptr<MechanicalState> getMechanicalState();
	void setMechanicalState(std::shared_ptr<MechanicalState> state);

	//bool addModule(std::string name, Module* module);

	bool addModule(std::shared_ptr<Module> module);

	template<class TModule>
	bool addModule(std::shared_ptr<TModule> tModule)
	{
		std::shared_ptr<Module> module = std::dynamic_pointer_cast<Module>(tModule);
		return addModule(module);
	}

	template<class TModule>
	bool deleteModule(std::shared_ptr<TModule> tModule)
	{
		std::shared_ptr<Module> module = std::dynamic_pointer_cast<Module>(tModule);
		return deleteModule(module);
	}

	bool deleteModule(std::shared_ptr<Module> module);

	void traverseBottomUp(Action* act);
	template<class Act>
	void traverseBottomUp() {
		Act action;
		doTraverseBottomUp(&action);
	}

	void traverseTopDown(Action* act);
	template<class Act>
	void traverseTopDown() {
		Act action;
		doTraverseTopDown(&action);
	}

	virtual void setAsCurrentContext();

	void setTopologyModule(std::shared_ptr<TopologyModule> topology);
	void setNumericalModel(std::shared_ptr<NumericalModel> numerical);
	void setCollidableObject(std::shared_ptr<CollidableObject> collidable);
	void setRenderController(std::shared_ptr<RenderController> controller);
	void setAnimationController(std::shared_ptr<AnimationController> controller);

	std::shared_ptr<CollidableObject>		getCollidableObject() { return m_collidable_object; }
	std::shared_ptr<NumericalModel>			getNumericalModel() { return m_numerical_model; }
	std::shared_ptr<TopologyModule>			getTopologyModule() { return m_topology; }
	std::shared_ptr<RenderController>		getRenderController() { return m_render_controller; }
	std::shared_ptr<AnimationController>	getAnimationController();

	//Module* getModule(std::string name);

	template<class TModule>
	std::shared_ptr<TModule> getModule()
	{
		TModule* tmp = new TModule;
		std::shared_ptr<Module> base;
		std::list<std::shared_ptr<Module>>::iterator iter;
		for (iter = m_module_list.begin(); iter != m_module_list.end(); iter++)
		{
			if ((*iter)->getClassInfo() == tmp->getClassInfo())
			{
				base = *iter;
				break;
			}
		}
		delete tmp;
		return TypeInfo::CastPointerDown<TModule>(base);
	}

	std::list<std::shared_ptr<Module>>& getModuleList() { return m_module_list; }

#define NODE_ADD_SPECIAL_MODULE( CLASSNAME, SEQUENCENAME ) \
	virtual void add##CLASSNAME( std::shared_ptr<CLASSNAME> module) { SEQUENCENAME.push_back(module); addModule(module);} \
	virtual void delete##CLASSNAME( std::shared_ptr<CLASSNAME> module) { SEQUENCENAME.remove(module); deleteModule(module); } \
	std::list<std::shared_ptr<CLASSNAME>>& get##CLASSNAME##List(){ return SEQUENCENAME;}

	NODE_ADD_SPECIAL_MODULE(ForceModule, m_force_list)
	NODE_ADD_SPECIAL_MODULE(ConstraintModule, m_constraint_list)
	NODE_ADD_SPECIAL_MODULE(CollisionModel, m_collision_list)
	NODE_ADD_SPECIAL_MODULE(VisualModule, m_render_list)
	NODE_ADD_SPECIAL_MODULE(TopologyMapping, m_topology_mapping_list)

	virtual bool initialize() { return true; }
	virtual void draw() {};
	virtual void advance(Real dt);
	virtual void takeOneFrame() {};
	virtual void updateModules() {};
	virtual void updateTopology() {};
	virtual bool resetStatus() { return true; }

protected:
	void setParent(Node* p) { m_parent = p; }

	virtual void doTraverseBottomUp(Action* act);
	virtual void doTraverseTopDown(Action* act);

private:
	Real m_dt;
	Real m_gravity = -9.8;
	bool m_initalized;

	VarField<Real> m_mass;
	HostVarField<bool> m_active;
	HostVarField<bool> m_visible;
	HostVarField<Real> m_time;

	HostVarField<std::string> m_node_name;

	std::list<std::shared_ptr<Module>> m_module_list;
	//std::map<std::string, Module*> m_modules;

	std::shared_ptr<TopologyModule> m_topology;
	std::shared_ptr<NumericalModel> m_numerical_model;
	std::shared_ptr<MechanicalState> m_mechanical_state;

	std::shared_ptr<CollidableObject> m_collidable_object;

	std::shared_ptr<RenderController> m_render_controller;
	std::shared_ptr<AnimationController> m_animation_controller;

	std::list<std::shared_ptr<ForceModule>> m_force_list;
	std::list<std::shared_ptr<ConstraintModule>> m_constraint_list;
	std::list<std::shared_ptr<CollisionModel>> m_collision_list;
	std::list<std::shared_ptr<VisualModule>> m_render_list;
	std::list<std::shared_ptr<TopologyMapping>> m_topology_mapping_list;

	std::shared_ptr<DeviceContext> m_context;

	ListPtr<Node> m_children;

	Node* m_parent;
};
}