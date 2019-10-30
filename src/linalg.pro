TARGET = linalgtest

TEMPLATE = app

CONFIG += c++2a

CONFIG -= app_bundle qt

macx {
    INCLUDEPATH += /usr/local/include

    QMAKE_CXXFLAGS += -fsanitize=address
    QMAKE_LFLAGS += -fsanitize=address
}

SOURCES += \
    main.cpp \
    matrix.cpp \
    matrix_operations.cpp \
    vector.cpp

HEADERS += \
    doctest.h \
    include/mat.h \
    include/mat_detail.h \
    include/mat_operations.h \
    include/vec.h \
    include/vec_detail.h \
    include/vec_trig.h
