/*
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
 */

import * as React from 'react';
import { Route, Switch, Router, Redirect } from 'react-router-dom';

import { history, AppRoute } from '../../app/AppNavigator';
import LoginPage from '../../layouts/LoginPage';
import DashboardLayout from '../../layouts/Admin';

class AppRouter extends React.Component {

  componentDidMount() {
    const { appUtils } = this.props;

    const user = appUtils.rafikiClient.getCurrentUser();
    if (!user) {
      appUtils.appNavigator.goTo(AppRoute.LOGIN);
    }
  }

  render() {
    const { appUtils } = this.props;

    return (
      <Router history={history}>
        <Switch>
          <Route exact path={AppRoute.LOGIN} render={(props) => <LoginPage {...props} appUtils={appUtils} />}/>
          <Route path={AppRoute.DASHBOARD} render={(props) => <DashboardLayout {...props} appUtils={appUtils} />}/>
          <Redirect to={AppRoute.DATASETS} />
        </Switch>
      </Router>
    );
  }
}

export default AppRouter;