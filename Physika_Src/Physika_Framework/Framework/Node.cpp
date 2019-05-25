#include "Node.h"
#include "Physika_Framework/Action/Action.h"

namespace Physika
{
IMPLEMENT_CLASS(Node)

Node::Node(std::string name)
	: Base()
	, m_parent(NULL)
{
	attachField(&m_active, "active", "this is a variable!", false);
	attachField(&m_visible, "visible", "this is a variable!", false);
	attachField(&m_time, "time", "this is a variable!", false);
	attachField(&m_node_name, "node_name", "Node name", false);

	m_active.setValue(true);
	m_visible.setValue(true);
	m_time.setValue(0.0f);
	m_node_name.setValue(name);

// 	m_active = HostVarField<bool>::createField(this, "active", "this is a variable!", true);
// 	m_visible = HostVarField<bool>::createField(this, "visible", "this is a variable!", true);
// 	m_time = HostVarField<float>::createField(this, "time", "this is a variable!", 0.0f);
	setName(name);

	m_mass.setValue(1.0);
	m_dt = 0.001f;
}


Node::~Node()
{
}

void Node::setName(std::string name)
{
	m_node_name.setValue(name);
}

std::string Node::getName()
{
	return m_node_name.getValue();
}


Node* Node::getChild(std::string name)
{
	for (ListPtr<Node>::iterator it = m_children.begin(); it != m_children.end(); ++it)
	{
		if ((*it)->getName() == name)
			return it->get();
	}
	return NULL;
}

Node* Node::getParent()
{
	return m_parent;
}

Node* Node::getRoot()
{
	Node* root = this;
	while (root->getParent() != NULL)
	{
		root = root->getParent();
	}
	return root;
}

bool Node::isActive()
{
	return m_active.getValue();
}

void Node::setActive(bool active)
{
	m_active.setValue(active);
}

bool Node::isVisible()
{
	return m_visible.getValue();
}

void Node::setVisible(bool visible)
{
	m_visible.setValue(visible);
}

float Node::getTime()
{
	return m_time.getValue();
}

float Node::getDt()
{
	return m_dt;
}

void Node::setDt(Real dt)
{
	m_dt = dt;
}

void Node::setGravity(Real g)
{
	m_gravity = g;
}

Real Node::getGravity()
{
	return m_gravity;
}

void Node::setMass(Real mass)
{
	m_mass.setValue(mass);
}

Real Node::getMass()
{
	return m_mass.getValue();
}

void Node::removeChild(std::shared_ptr<Node> child)
{
	ListPtr<Node>::iterator iter = m_children.begin();
	for (; iter != m_children.end(); )
	{
		if (*iter == child)
		{
			m_children.erase(iter++);
		}
		else
		{
			++iter;
		}
	}
}

void Node::advance(Real dt)
{
	auto nModel = this->getNumericalModel();
	if (nModel == NULL)
	{
		Log::sendMessage(Log::Warning, this->getName() + ": No numerical model is set!");
	}
	else
	{
		nModel->step(this->getDt());
	}
}

std::shared_ptr<DeviceContext> Node::getContext()
{
	if (m_context == nullptr)
	{
		m_context = TypeInfo::New<DeviceContext>();
		m_context->setParent(this);
		addModule(m_context);
	}
	return m_context;
}

void Node::setContext(std::shared_ptr<DeviceContext> context)
{
	if (m_context != nullptr)
	{
		deleteModule(m_context);
	}

	m_context = context; 
	addModule(m_context);
}

std::shared_ptr<MechanicalState> Node::getMechanicalState()
{
	if (m_mechanical_state == nullptr)
	{
		m_mechanical_state = TypeInfo::New<MechanicalState>();
		m_mechanical_state->setParent(this);
		addModule(m_mechanical_state);
	}
	return m_mechanical_state;
}

void Node::setMechanicalState(std::shared_ptr<MechanicalState> state)
{
	if (m_mechanical_state != nullptr)
	{
		deleteModule(m_mechanical_state);
	}

	m_mechanical_state = state; 
	addModule(state);
}

/*
std::shared_ptr<MechanicalState> Node::getMechanicalState()
{
	if (m_mechanical_state == nullptr)
	{
		m_mechanical_state = TypeInfo::New<MechanicalState>();
		m_mechanical_state->setParent(this);
	}
	return m_mechanical_state;
}*/
/*
bool Node::addModule(std::string name, Module* module)
{
	if (getContext() == nullptr || module == NULL)
	{
		std::cout << "Context or module does not exist!" << std::endl;
		return false;
	}

	std::map<std::string, Module*>::iterator found = m_modules.find(name);
	if (found != m_modules.end())
	{
		std::cout << "Module name already exists!" << std::endl;
		return false;
	}
	else
	{
		m_modules[name] = module;
		m_module_list.push_back(module);

//		module->insertToNode(this);
	}

	return true;
}
*/
bool Node::addModule(std::shared_ptr<Module> module)
{
	auto found = std::find(m_module_list.begin(), m_module_list.end(), module);
	if (found == m_module_list.end())
	{
		m_module_list.push_back(module);
		module->setParent(this);
		return true;
	}

	return false;
}

bool Node::deleteModule(std::shared_ptr<Module> module)
{
	auto found = std::find(m_module_list.begin(), m_module_list.end(), module);
	if (found != m_module_list.end())
	{
		m_module_list.erase(found);
		return true;
	}
		
	return true;
}

void Node::doTraverseBottomUp(Action* act)
{
	ListPtr<Node>::iterator iter = m_children.begin();
	for (; iter != m_children.end(); iter++)
	{
		(*iter)->traverseBottomUp(act);
	}

	act->Process(this);
}

void Node::doTraverseTopDown(Action* act)
{
	act->Process(this);

	ListPtr<Node>::iterator iter = m_children.begin();
	for (; iter != m_children.end(); iter++)
	{
		(*iter)->traverseBottomUp(act);
	}
}

void Node::traverseBottomUp(Action* act)
{
	doTraverseBottomUp(act);
}

void Node::traverseTopDown(Action* act)
{
	doTraverseTopDown(act);
}

void Node::setAsCurrentContext()
{
	getContext()->enable();
}

void Node::setTopologyModule(std::shared_ptr<TopologyModule> topology)
{
	if (m_topology != nullptr)
	{
		deleteModule(m_topology);
	}
	m_topology = topology;
	addModule(topology);
}

void Node::setNumericalModel(std::shared_ptr<NumericalModel> numerical)
{
	if (m_numerical_model != nullptr)
	{
		deleteModule(m_numerical_model);
	}
	m_numerical_model = numerical;
	addModule(numerical);
}

void Node::setCollidableObject(std::shared_ptr<CollidableObject> collidable)
{
	if (m_collidable_object != nullptr)
	{
		deleteModule(m_collidable_object);
	}
	m_collidable_object = collidable;
	addModule(collidable);
}

void Node::setRenderController(std::shared_ptr<RenderController> controller)
{
	if (m_render_controller != nullptr)
	{
		deleteModule(m_render_controller);
	}
	m_render_controller = controller;
	addModule(m_render_controller);
}

void Node::setAnimationController(std::shared_ptr<AnimationController> controller)
{
	if (m_animation_controller != nullptr)
	{
		deleteModule(m_animation_controller);
	}
	m_animation_controller = controller;
	addModule(m_animation_controller);
}

std::shared_ptr<AnimationController> Node::getAnimationController()
{
	return m_animation_controller;
}

/*Module* Node::getModule(std::string name)
{
	std::map<std::string, Module*>::iterator result = m_modules.find(name);
	if (result == m_modules.end())
	{
		return NULL;
	}

	return result->second;
}*/

}