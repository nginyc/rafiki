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
import CssBaseline from '@material-ui/core/CssBaseline';

import RafikiClient from '../client/RafikiClient';
import AppNavigator from './app/AppNavigator';
import AppRouter from './components/app/AppRouter';
import AppAlert, { AppAlertOption, AppAlertType } from './components/app/AppAlert';

interface AppUtils {
  showError: (error: Error, title?: string) => void;
  showSuccess: (message: string, title?: string) => void;
  rafikiClient: RafikiClient;
  appNavigator: AppNavigator;
}

declare global {
  interface Window { 
    ADMIN_HOST: string;
    ADMIN_PORT: number;
  }
}

class App extends React.Component {
  appUtils: AppUtils;

  state: {
    appAlertOption?: AppAlertOption;
    appAlertIsOpen: boolean;
  } = {
    appAlertOption: null,
    appAlertIsOpen: false
  }

  constructor(props: {}) {
    super(props);

    const adminHost = window.ADMIN_HOST || 'localhost'; 
    const adminPort = window.ADMIN_PORT || 3000;
    
    const rafikiClient = new RafikiClient(adminHost, adminPort);
    const appNavigator = new AppNavigator();

    this.appUtils = {
      showError: (...x) => this.showError(...x),
      showSuccess: (...x) => this.showSuccess(...x),
      rafikiClient,
      appNavigator
    };
  }

  showSuccess(message: string, title: string = 'Success!') {
    this.setState({
      appAlertOption: {
        title,
        message,
        type: AppAlertType.SUCCESS
      },
      appAlertIsOpen: true
    })
  }

  showError(error: Error, title: string = 'An Error Occured') {
    
    console.error(error);

    this.setState({ 
      appAlertOption: {
        title,
        message: error.message,
        type: AppAlertType.ERROR
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
export { AppUtils };
