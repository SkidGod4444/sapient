// The SDK is a `file:` dependency living outside this app's root
// (../../sdks/typescript) — Metro needs the real path in watchFolders to
// resolve the symlink.
const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

const config = getDefaultConfig(__dirname);
const sdkPath = path.resolve(__dirname, '../../sdks/typescript');

config.watchFolders = [...(config.watchFolders ?? []), sdkPath];
config.resolver.extraNodeModules = {
  ...(config.resolver.extraNodeModules ?? {}),
  '@openhorizon/sapient': sdkPath,
};
// Modules imported from the watched SDK folder (e.g. injected @babel/runtime
// helpers) must resolve against this app's node_modules.
config.resolver.nodeModulesPaths = [
  path.resolve(__dirname, 'node_modules'),
  ...(config.resolver.nodeModulesPaths ?? []),
];

module.exports = config;
