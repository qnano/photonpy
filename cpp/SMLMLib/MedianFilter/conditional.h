// Copyright (c) 2018 sergiy.yevtushenko(at)gmail.com under <http://www.opensource.org/licenses/mit-license>

#pragma once

namespace siy {
    template<bool B, class T, class F>
    struct conditional { typedef T type; };

    template<class T, class F>
    struct conditional<false, T, F> { typedef F type; };
}
