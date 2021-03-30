#pragma once

#include <cstdint>

#pragma pack(push, 4)

template<typename T>
struct vector2 {
	vector2() { x = y = 0.0; }
	template<typename Tx, typename Ty>
	vector2(Tx X, Ty Y) { x = (T)X; y = (T)Y; }
	T x, y;
};

typedef vector2<float> vector2f;
typedef vector2<double> vector2d;

template<typename T>
struct vector3 {
	vector3() { x = y = z = 0.0f; }
	template<typename Tx, typename Ty, typename Tz>
	vector3(Tx X, Ty Y, Tz Z) { x = (T)X; y = (T)Y; z = (T)Z; }
	template<typename Tc>
	vector3(const vector3<Tc>& o) : x(o.x), y(o.y), z(o.z) {}
	T x, y, z;

	vector3 operator*(const vector3& o) const {
		return vector3(x*o.x, y*o.y, z*o.z);
	}
	vector3 operator*(T a) const {
		return vector3(x*a, y*a, z*a);
	}
	friend vector3 operator*(T a, const vector3& b) {
		return b * a;
	}
	vector3 operator+(const vector3& o) const {
		return vector3(x + o.x, y + o.y, z + o.z);
	}
	vector3& operator+=(const vector3& o) {
		x += o.x; y += o.y; z += o.z; return *this;
	}
	vector3& operator-=(const vector3& o) {
		x -= o.x; y -= o.y; z -= o.z; return *this;
	}
	vector3 operator-(const vector3& o) const {
		return vector3(x - o.x, y - o.y, z - o.z);
	}
	vector3 operator-(float a) const {
		return vector3(x - a, y - a, z - a);
	}
	vector3 operator+(float a) const {
		return vector3(x + a, y + a, z + a);
	}
	vector3 operator-() const {
		return vector3(-x, -y, -z);
	}
	vector3& operator*=(const vector3& o) {
		x *= o.x; y *= o.y; z *= o.z;
		return *this;
	}
	vector3& operator*=(T a) {
		x *= a; y *= a; z *= a;
		return *this;
	}
	vector3& operator/=(T a) {
		x /= a; y /= a; z /= a;
		return *this;
	}
	vector3 operator/(T a) {
		return vector3(x / a, y / a, z / a);
	}
	T length() {
		return sqrtf(x*x + y * y + z * z);
	}
	template<typename T>
	friend vector3 operator/(T a, vector3<T> b) {
		return vector3<T>(a / b.x, a / b.y, a / b.z);
	}

	vector2<T> xy()
	{
		return vector2<T>(x, y);
	}

};

template<typename T>
inline vector3<T> sqrt(const vector3<T>& a) { return vector3<T>(sqrt(a.x), sqrt(a.y), sqrt(a.z)); }

typedef vector3<float> vector3f;
typedef vector3<double> vector3d;

#pragma pack(pop)

#define MATH_PI (3.141592653589793f)

template<typename T>
constexpr T Factorial(T n) {
	return n == 0 ? 1 : n * Factorial(n - 1);
}

template<typename T>
constexpr T IntegerPower(T val, int e) {
	return e == 0 ? 1 : IntegerPower(val, e - 1)*val;
}



typedef uint32_t uint;
