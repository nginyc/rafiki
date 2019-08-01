//axios to send ajax request
import axios from 'axios'
import HTTPconfig from "../HTTPconfig"

export const requestSignIn = (authData) => {
  console.log("requestSignIn", `${HTTPconfig.gateway}tokens`)
  return axios({
    method: 'post',
    url: `${HTTPconfig.gateway}tokens`,
    headers: HTTPconfig.HTTP_HEADER,
    // `auth` indicates that HTTP Basic auth should be used, and supplies credentials.
    // This will set an `Authorization` header, overwriting any existing
    // `Authorization` custom headers you have set using `headers`.
    // Please note that only HTTP Basic auth is configurable through this parameter.
    // For Bearer tokens and such, use `Authorization` custom headers instead.
    data: JSON.stringify(authData)
  });
}