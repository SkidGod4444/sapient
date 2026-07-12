#import <Foundation/Foundation.h>
#import "SapientReactNativeImpl.h"
#import <ReactCommon/CxxTurboModuleUtils.h>

@interface SapientReactNativeOnLoad : NSObject
@end

@implementation SapientReactNativeOnLoad

using namespace facebook::react;

+ (void)load
{
  registerCxxModuleToGlobalModuleMap(
    std::string(SapientReactNativeImpl::kModuleName),
    [](std::shared_ptr<CallInvoker> jsInvoker) {
      return std::make_shared<SapientReactNativeImpl>(jsInvoker);
    }
  );
}

@end
