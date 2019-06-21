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