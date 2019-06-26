import * as React from 'react';
import CssBaseline from '@material-ui/core/CssBaseline';

import RafikiClient from './client/RafikiClient';
import AppNavigator from './app/AppNavigator';
import PlotManager from './app/PlotManager';
import AppRouter from './components/app/AppRouter';
import AppAlert from './components/app/AppAlert';

class App extends React.Component {
  state = {
    appAlertOption: null,
    appAlertIsOpen: false
  }

  constructor(props) {
    super(props);


    const adminHost = window.ADMIN_HOST || "localhost"; 
    const adminPort = window.ADMIN_PORT || 3000;

    const rafikiClient = new RafikiClient(adminHost, adminPort);
    const appNavigator = new AppNavigator();
    const plotManager = new PlotManager();

    this.appUtils = {
      showError: (...x) => this.showError(...x),
      showSuccess: (...x) => this.showSuccess(...x),
      rafikiClient,
      appNavigator,
      plotManager
    };
  }

  showSuccess(message, title = 'Success!') {
    this.setState({
      appAlertOption: {
        title,
        message,
        type: "SUCCESS"
      },
      appAlertIsOpen: true
    })
  }

  showError(error, title = 'An Error Occured') {
    
    console.error(error);

    this.setState({ 
      appAlertOption: {
        title,
        message: error.message,
        type: "ERROR" 
      }, 
      appAlertIsOpen: true
    });
  }

  render() {
    const { appAlertIsOpen, appAlertOption } = this.state;

    return (
      <React.Fragment>
        <CssBaseline />
        <AppRouter appUtils={this.appUtils} />
        <AppAlert
          isOpen={appAlertIsOpen} 
          option={appAlertOption}
          onClose={() => {
            this.setState({ appAlertIsOpen: false });
          }}
        />
      </React.Fragment>
    );
  }
}

export default App;
