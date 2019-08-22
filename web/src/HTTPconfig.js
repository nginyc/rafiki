// start script's process.env.NODE_ENV = 'development';
// build script's process.env.NODE_ENV = 'production';
// default as development

const adminHost = process.env.REACT_APP_API_POINT_HOST
const adminPort = process.env.REACT_APP_API_POINT_PORT


const HTTPconfig = {
  // the client tells server data-type json is actually sent.
  HTTP_HEADER: {
    "Content-Type": "application/json",
  },
  UPLOAD_FILE: {
    'Content-Type':'multipart/form-data',
  },
  // need a working server for axios uploadprogress to work
  // gateway: "http://localhost:5000/",
  // gateway: "http://ncrs.d2.comp.nus.edu.sg:3000/"
  adminHost: `${adminHost}`,
  adminPort: `${adminPort}`,
  gateway: `http://${adminHost}:${adminPort}/`
}

if (process.env.NODE_ENV === "production") {
  //HTTPconfig.gateway = "http://13.229.126.135/"
  HTTPconfig.gateway = `http://${adminHost}:${adminPort}/`
}

export default HTTPconfig;
