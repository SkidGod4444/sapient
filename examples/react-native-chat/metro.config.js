// The SDK is a `file:` dependency living outside this app's root
// (../../sdks/typescript) — Metro needs the real path in watchFolders to
// resolve the symlink.
const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

const config = getDefaultConfig(__dirname);
const sdkPath = path.resolve(__dirname, '../../sdks/typescript');
const nativePath = path.resolve(__dirname, '../../sdks/react-native');

config.watchFolders = [...(config.watchFolders ?? []), sdkPath, nativePath];
config.resolver.extraNodeModules = {
  ...(config.resolver.extraNodeModules ?? {}),
  '@openhorizon-labs/sapient': sdkPath,
  '@openhorizon-labs/sapient-react-native': nativePath,
};
// Modules imported from the watched SDK folder (e.g. injected @babel/runtime
// helpers) must resolve against this app's node_modules.
config.resolver.nodeModulesPaths = [
  path.resolve(__dirname, 'node_modules'),
  ...(config.resolver.nodeModulesPaths ?? []),
];


// The native package's own node_modules carries a NEWER react-native (dev
// dependency of the library scaffold) whose Flow sources Metro can't parse
// (`match` expressions) — and duplicated react/react-native breaks apps
// anyway. Resolve those two only from THIS app's node_modules; everything
// else in the library's node_modules (e.g. uniffi-bindgen-react-native's
// runtime) stays resolvable.
// metro-config only exports `./private/*` now — the old
// `metro-config/src/defaults/exclusionList` deep import throws
// ERR_PACKAGE_PATH_NOT_EXPORTED on the Metro that ships with Expo SDK 54 —
// and the module is ESM-interop, so the function hangs off `.default`.
const exclusionListModule = require('metro-config/private/defaults/exclusionList');
const exclusionList = exclusionListModule.default ?? exclusionListModule;
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
config.resolver.blockList = exclusionList([
  new RegExp(`${escapeRe(nativePath)}/node_modules/react-native/.*`),
  new RegExp(`${escapeRe(nativePath)}/node_modules/react/.*`),
]);

module.exports = config;
