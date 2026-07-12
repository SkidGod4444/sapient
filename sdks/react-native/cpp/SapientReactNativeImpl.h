#pragma once

#include <SapientReactNativeSpecJSI.h>

#include <memory>

namespace facebook::react {

class SapientReactNativeImpl
  : public NativeSapientReactNativeCxxSpec<SapientReactNativeImpl> {
public:
  SapientReactNativeImpl(std::shared_ptr<CallInvoker> jsInvoker);

  double multiply(jsi::Runtime& rt, double a, double b);
};

}
