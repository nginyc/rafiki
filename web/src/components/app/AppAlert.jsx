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
import { withStyles } from '@material-ui/core/styles';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  Snackbar
} from '@material-ui/core';

const styles = (theme) => ({});

/* export enum AppAlertType { ERROR, SUCCESS } */

class AppAlert extends React.Component {

  renderDialog() {
    const { option, isOpen, onClose } = this.props;

    if (!option) {
      return;
    }
    
    return (
      <Dialog open={(option.type === "ERROR") && isOpen} 
        onClose={onClose}>
        <DialogTitle>{option.title || 'Alert'}</DialogTitle>
        {option.message && 
          <DialogContent>
            <DialogContentText>{option.message}</DialogContentText>
          </DialogContent>
        }
        <DialogActions>
          <Button onClick={onClose} color="primary" autoFocus>
            Okay
          </Button>
        </DialogActions>
      </Dialog>
    );
  }

  renderSnackbar() {
    const { option, isOpen, onClose } = this.props;

    if (!option) {
      return;
    }

    return (
      <Snackbar
        open={option.type === "SUCCESS" && isOpen}
        onClose={onClose}
        autoHideDuration={1000}
        message={<span>{option.title}: {option.message}</span>}
      />
    );
  }

  render() {
    return (
      <React.Fragment>
        {this.renderDialog()}
        {this.renderSnackbar()}
      </React.Fragment>
    );
  }
}

export default withStyles(styles)(AppAlert);