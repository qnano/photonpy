#pragma once


#define PLL_LAMBDA [=]__device__ __host__
#define PLLFN __device__ __host__

template<typename T, int D>
struct Indexer
{
	T* data;
	Vector<int, D> pitch;

	Indexer(const std::pair<T*, Vector<int, D>>& x) : data(x.first), pitch(x.second) {}

	PLLFN Indexer<T, D - 1> operator[](int idx) const {
		return Indexer<T, D - 1>{&data[pitch[0] * idx], pitch.template slice<D - 1>(1)};
	}
};

template<typename T>
struct Indexer<T, 1>
{
	T* data;
	Vector<int, 1> pitch;
	PLLFN T& operator[](int idx) const {
		return data[pitch[0] * idx];
	}
};

template<typename T, int D>
class Output : public Indexer<T, D>
{
public:
//	Output(Binding& db, const char* name, std::initializer_list<int> shape, ParamFlags flags = None)
	//	: Indexer<T, D>(db.Register<T, D>(name, shape, ParamFlags(flags + IsOutput))) { }
};


template<typename T, int D>
struct ConstIndexer
{
	const T* data;
	Vector<int, D> pitch;

	ConstIndexer(const std::pair<T*, Vector<int, D>>& x) : data(x.first), pitch(x.second) {}

	PLLFN ConstIndexer<T, D - 1> operator[](int idx) const {
		return ConstIndexer<T, D - 1>{&data[pitch[0] * idx], pitch.template slice<D - 1>(1)};
	}
};

template<typename T>
struct ConstIndexer<T, 1>
{
	const T* data;
	Vector<int, 1> pitch;

	PLLFN const T& operator[] (int idx) const {
		return data[idx * pitch[0]];
	}
};

template<typename T, int D>
class Input : public ConstIndexer<T, D>
{
public:
	//Input(Binding& db, const char* name, std::initializer_list<int> shape, ParamFlags flags = Required)
		//: ConstIndexer<T, D>(db.Register<T, D>(name, shape, (ParamFlags)(flags + IsInput))) {}
};

template<typename T, int D>
class Temporary : public Indexer<T, D>
{
public:
	//Temporary(Binding& db, const char* name, std::initializer_list<int> shape) :
		//Indexer<T, D>(db.Register<T, D>(name, shape, None)) {}
};

template<typename T, int D>
class InOut : public Indexer<T, D>
{
public:
	//InOut(Binding& db, const char* name, std::initializer_list<int> shape, ParamFlags flags = Required) :
		//Indexer<T, D>(db.Register<T, D>(name, shape, ParamFlags(flags + IsInput + IsOutput))) { }
};

