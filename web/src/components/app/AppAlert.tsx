import * as React from 'react';
import { withStyles, StyleRulesCallback } from '@material-ui/core/styles';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  Snackbar
} from '@material-ui/core';

const styles: StyleRulesCallback = (theme) => ({});

enum AppAlertType { ERROR, SUCCESS }

interface AppAlertOption {
  title: string;
  message: string;
  type: AppAlertType;
}

interface Props {
  isOpen: boolean;
  option: AppAlertOption | null;
  onClose: () => void;
}

class AppAlert extends React.Component<Props> {

  renderDialog() {
    const { option, isOpen, onClose } = this.props;

    if (!option) {
      return;
    }
    
    return (
      <Dialog open={(option.type == AppAlertType.ERROR) && isOpen} 
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
        open={option.type == AppAlertType.SUCCESS && isOpen}
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
export { AppAlertType, AppAlertOption }