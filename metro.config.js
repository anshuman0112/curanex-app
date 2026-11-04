const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
// The backend holds gigabytes of model repos and datasets; the app never imports from it.
config.resolver.blockList = [new RegExp('[/\\\\]curanex-app[/\\\\]backend[/\\\\].*')];
module.exports = config;
