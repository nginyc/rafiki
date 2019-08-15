import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import { compose } from "redux";
import { Link, withRouter } from "react-router-dom";

import { withStyles } from '@material-ui/core/styles';
import Divider from '@material-ui/core/Divider';
import Drawer from '@material-ui/core/Drawer';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';

// for icons
import DnsRoundedIcon from '@material-ui/icons/DnsRounded';
// row-based table (dataset)
import ListIcon from '@material-ui/icons/FormatListBulleted'
import CloudUpload from '@material-ui/icons/CloudUploadOutlined'

// for nested list
import Collapse from '@material-ui/core/Collapse';
import ExpandLess from '@material-ui/icons/ExpandLess';
import ExpandMore from '@material-ui/icons/ExpandMore';

import Logo from "assets/Logo-Rafiki-cleaned.png"

// customize scrollbar for the fixed-div navigator
import SimpleBar from 'simplebar-react';
import 'simplebar/dist/simplebar.min.css';

// Navigator basic color dark blue specified in
// ConsoleTheme MuiDrawer's paper
const styles = theme => ({
  categoryHeader: {
    paddingTop: 16,
    paddingBottom: 16,
  },
  categoryHeaderPrimary: {
    color: theme.palette.common.white,
  },
  categoryHeaderPrimaryActive: {
    color: 'inherit'
  },
  item: {
    paddingTop: 11,
    paddingBottom: 11,
    color: 'rgba(255, 255, 255, 0.7)',
  },
  itemCategory: {
    backgroundColor: '#232f3e',
    boxShadow: '0 -1px 0 #404854 inset',
    paddingTop: 16,
    paddingBottom: 16,
  },
  firebase: {
    fontSize: 24,
    fontFamily: theme.typography.fontFamily,
    color: theme.palette.common.white,
  },
  logo: {
    height: 28,
    marginRight: 10
  },
  logoText: {
    // color: "#61ADB1 "
  },
  itemActionable: {
    '&:hover': {
      backgroundColor: 'rgba(255, 255, 255, 0.08)',
    },
  },
  overviewHover: {
    '&:hover': {
      backgroundColor: 'rgba(216, 255, 255, 0.1)',
    },
  },
  itemActiveItem: {
    color: theme.palette.secondary.main,
  },
  itemPrimary: {
    color: 'inherit',
    fontSize: theme.typography.fontSize,
  },
  divider: {
    marginTop: theme.spacing(1) * 2,
  }
});


class Navigator extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    location: PropTypes.object,
  }

  state = {
    DatasetsTableOpen: true,
    JobsTableOpen: false,
    DataApplicationOpen: false,
    KeyValueDBOpen: false
  };

  handleClick = (categoryHeader) => {
    switch(categoryHeader) {
      case "Datasets":
        this.setState(state => (
          { DatasetsTableOpen: !state.DatasetsTableOpen }
        ));
        break
      case "Jobs":
        this.setState(state => (
          { JobsTableOpen: !state.JobsTableOpen }
        ));
        break
      case "Application":
        this.setState(state => (
          { DataApplicationOpen: !state.DataApplicationOpen }
        ));
        break
      case "KeyValue":
        this.setState(state => (
          { KeyValueDBOpen: !state.KeyValueDBOpen }
        ));
        break
      default:
        this.setState(state => (
          { JobsTableOpen: !state.DatasetsTableOpen }
        ));
        return
    }
  };

  render() {
    const categories = [
      {
        id: 'Dataset',
        collapseID: "Datasets",
        collapseIn: this.state.DatasetsTableOpen,
        children: [
          {
            id: 'List Dataset',
            icon: <ListIcon />,
            pathname: "/console/datasets/list-dataset"
          },
          {
            id: 'Upload Dataset',
            icon: <CloudUpload />,
            pathname: "/console/datasets/upload-datasets"
          },
          // {
          //   id: 'Delete Dataset',
          //   icon: <DeleteDsIcon />,
          //   pathname: "/console/datasets/delete-dataset"
          // },
        ],
      },
      {
        id: 'Training Jobs',
        collapseID: "Jobs",
        collapseIn: this.state.JobsTableOpen,
        children: [
          {
            id: 'List Train Jobs',
            icon: <ListIcon />,
            pathname: "/console/jobs/list-train-jobs"
          },
          {
            id: 'Create NEW Train',
            icon: <CloudUpload />,
            pathname: "/console/jobs/create-train-job"
          },
        ],
      },
      {
        id: 'Applications',
        collapseID: "Application",
        collapseIn: this.state.DataApplicationOpen,
        children: [
          {
            id: 'Applications',
            icon: <DnsRoundedIcon />,
            pathname: "/console/application/list-applications"
          }
        ],
      },
    ];

    const {
      classes,
      location,
      staticContext,
      open,
      onClose,
      ...other
    } = this.props;

    return (
      <Drawer
        variant="permanent"
        open={open}
        onClose={onClose}
        {...other}
      >
        <SimpleBar style={{width: 255}}>
          <List disablePadding>
            <ListItem
              component={Link}
              to="/"
              className={classNames(
                classes.firebase,
                classes.item,
                classes.itemCategory)}
            >
              <img alt="logo" src={Logo} className={classes.logo} />
              <span className={classes.logoText}>Rafiki</span>
            </ListItem>
         
            {categories.map(({id, collapseID, collapseIn, children }) => (
              <React.Fragment key={id}>
                <ListItem
                  button
                  onClick={() => this.handleClick(collapseID)}
                  className={classes.categoryHeader}
                >
                  <ListItemText
                    classes={{
                      primary: classes.categoryHeaderPrimary,
                    }}
                  >
                    {id}
                  </ListItemText>
                  {collapseIn
                    ? <ExpandLess 
                        style={{
                          color: "white"
                        }}
                      />
                    : <ExpandMore
                        style={{
                          color: "white"
                        }}
                      />}
                </ListItem>
                <Collapse in={collapseIn} timeout="auto" unmountOnExit>
                  {children.map(({ id: childId, icon, pathname }) => (
                    <ListItem
                      key={childId}
                      button
                      onClick={onClose}
                      component={Link}
                      to={pathname}
                      dense
                      className={classNames(
                        classes.item,
                        classes.itemActionable,
                        location.pathname === pathname &&
                        classes.itemActiveItem,
                      )}
                    >
                      <ListItemIcon>{icon}</ListItemIcon>
                      <ListItemText
                        classes={{
                          primary: classes.itemPrimary,
                        }}
                      >
                        {childId}
                      </ListItemText>
                    </ListItem>
                  ))}
                </Collapse>
                <Divider className={classes.divider} />
              </React.Fragment>
            ))}
          </List>
        </SimpleBar>
      </Drawer>
    );
  }
}


export default compose(
  withRouter,
  withStyles(styles)
)(Navigator)
