const webpack = require('webpack');
const packageJson = require('./package.json');
const CleanWebpackPlugin = require('clean-webpack-plugin');

module.exports = {
  mode: 'development',
  devtool: 'source-map',
  entry: {
    app: [
      './src/index.tsx'
    ]
  },
  resolve: {
    extensions: [".ts", ".tsx", ".js", ".json"]
  },
  output: {
    path: __dirname + '/dist',
    filename: 'bundle.js',
    publicPath: __dirname + '/dist'
  },
  plugins: [
    new CleanWebpackPlugin(['dist']),
    new webpack.DefinePlugin({
      'window.ADMIN_HOST': JSON.stringify(process.env.RAFIKI_ADDR),
      'window.ADMIN_PORT': JSON.stringify(process.env.ADMIN_EXT_PORT)
    })
  ],
  module: {
    rules: [
      { 
        test: /\.tsx?$/, 
        loader: "awesome-typescript-loader",
        exclude: /node_modules/ 
      },
      { 
        test: /\.js?$/, 
        loader: "source-map-loader" 
      },
      {
        test: /\.(png|svg|jpg|gif)$/,
        use: ["file-loader"]
      },
      {
        test: /\.(woff|woff2|eot|ttf|otf)$/,
        use: ["file-loader"]
      }
    ]
  },
  // When importing a module whose path matches one of the following, just
  // assume a corresponding global variable exists and use that instead.
  // This is important because it allows us to avoid bundling all of our
  // dependencies, which allows browsers to cache those libraries between builds.
  externals: {
    "react": "React",
    "react-dom": "ReactDOM"
  },  
  devServer: {
    publicPath: '/dist/',
    historyApiFallback: true
  }
};
