// Context class to manage C++ objects lifetime from python
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "Context.h"
#include "ThreadUtils.h"
#include "cuda_runtime.h"

ContextObject::~ContextObject()
{
	if(context)
		context->Remove(this);
}

void ContextObject::SetContext(Context * ctx)
{
	if (context) {
		context->Remove(this);
		context = 0;
	}

	if (ctx) {
		ctx->Add(this);
	}
}

void ContextObject::SetCudaDevice()
{
	if (context)
		cudaSetDevice(context->GetDeviceIndex());
}

void Context::Remove(ContextObject * obj)
{
	LockedAction(mtx, [&]() {
		objects.erase(obj);
		obj->context = 0;
	});
}

Context::~Context()
{
	mtx.lock();
	// Go through all objects without an iterator that can be invalidated
	while (!objects.empty()) {
		auto first = objects.begin();
		ContextObject* obj = *first;
		objects.erase(first);

		mtx.unlock();
		delete obj;
		mtx.lock();
	}
	mtx.unlock();
}

void Context::Add(ContextObject * obj)
{
	LockedAction(mtx, [&]() {
		objects.insert(obj);
		obj->context = this;
	});
}

CDLL_EXPORT Context * Context_Create()
{
	return new Context();
}

CDLL_EXPORT void Context_Destroy(Context * ctx)
{
	delete ctx;
}
