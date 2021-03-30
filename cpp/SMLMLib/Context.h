// Context class to manage C++ objects lifetime from python
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "DLLMacros.h"
#include <unordered_set>
#include <mutex>

class Context;
class ContextObject;

class Context {
	std::mutex mtx;
	std::unordered_set<ContextObject*> objects;
public:
	Context(int deviceIndex=0) : deviceIndex(deviceIndex) {}
	DLL_EXPORT ~Context();

	int GetDeviceIndex() { return deviceIndex; }
private:
	int deviceIndex;

	DLL_EXPORT void Add(ContextObject* obj);
	DLL_EXPORT void Remove(ContextObject* obj);
	friend class ContextObject;
};

class DLL_EXPORT ContextObject {
public:
	ContextObject(Context*ctx=0) :context(0) { if(ctx) ctx->Add(this); }
	virtual ~ContextObject();
	void SetContext(Context* ctx);
	Context* GetContext() { return context; }
protected:
	void SetCudaDevice();
	Context* context;
	friend class Context;
};


CDLL_EXPORT Context* Context_Create();
CDLL_EXPORT void Context_Destroy(Context* ctx);

