import React, { Component } from 'react';
import {
  Switch,
  Route 
 } from 'react-router-dom'


 import Loadable from 'react-loadable';
 import LinearProgress from "@material-ui/core/LinearProgress";

import LandingPage from "./containers/LandingPage/LandingPage"
import PublicationsPage from "./containers/PublicationsPage/PublicationsPage"
import ContactPage from "./containers/ContactPage/ContactPage"

function Loading(props) {
  if (props.error) {
    return <div>Error! <button onClick={ props.retry }>Retry</button></div>;
  } else {
    return <LinearProgress color="secondary" />;
  }
}

const NoMatch = ({ location }) => (
  <h3>No page found for <code>{location.pathname}</code></h3>
)

const SignInLoadable = Loadable({
  loader: () => import("./containers/SignInPage/SignIn"),
  loading: Loading,
});

const SignUpLoadable = Loadable({
  loader: () => import("./containers/SignUpPage/SignUp"),
  loading: Loading,
});

const ConsoleAppFrameLoadable = Loadable({
  loader: () => import("./containers/ConsoleAppFrame/ConsoleAppFrame"),
  loading: Loading,
});

class App extends Component {
  render() {
    return (
        <Switch>
          <Route
            exact
            path='/'
            component={LandingPage}
          />
          <Route
            exact
            path='/publications'
            component={PublicationsPage}
          />
          <Route
            exact
            path='/contact'
            component={ContactPage}
          />
          <Route
            exact
            path='/sign-in'
            component={SignInLoadable}
          />
          <Route
            exact
            path='/sign-up'
            component={SignUpLoadable}
          />
          <Route
            path='/console'
            component={ConsoleAppFrameLoadable}
          />
          <Route component={NoMatch} />
        </Switch>
    );
  }
}

export default App;
