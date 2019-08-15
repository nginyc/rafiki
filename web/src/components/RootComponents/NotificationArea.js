import CloseIcon from "@material-ui/icons/Close";
import IconButton from "@material-ui/core/IconButton";
import PropTypes from "prop-types";
import React from "react";
import Snackbar from "@material-ui/core/Snackbar";

class NotificationArea extends React.PureComponent {
  static propTypes = {
    open: PropTypes.bool.isRequired,
    message: PropTypes.string.isRequired,
    handleClose: PropTypes.func.isRequired
  };

  render() {
    return (
      <Snackbar
        action={[
          <IconButton
            aria-label="Close"
            color="inherit"
            key="close"
            onClick={this.props.handleClose}
          >
            <CloseIcon />
          </IconButton>
        ]}
        anchorOrigin={{
          vertical: "bottom",
          horizontal: "left"
        }}
        ContentProps={{
          "aria-describedby": "message-id"
        }}
        message={<span>{this.props.message}</span>}
        open={this.props.open}
      />
    );
  }
}

export default NotificationArea;
