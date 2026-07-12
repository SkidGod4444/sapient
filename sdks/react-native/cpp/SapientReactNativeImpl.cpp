#include "SapientReactNativeImpl.h"

namespace facebook::react {

SapientReactNativeImpl::SapientReactNativeImpl(
  std::shared_ptr<CallInvoker> jsInvoker
)
  : NativeSapientReactNativeCxxSpec(std::move(jsInvoker)) {}

double SapientReactNativeImpl::multiply(
  jsi::Runtime& rt,
  double a,
  double b
) {
  return a * b;
}

}
