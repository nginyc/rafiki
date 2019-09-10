import React, { Fragment } from "react";
import PropTypes from "prop-types";

import withStyles from "@material-ui/core/styles/withStyles";

// table
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';

// table icons
import * as moment from 'moment';

const styles = theme => ({
  button: {
    margin: theme.spacing(1)
  },
  tableButtons : {
    margin: 3,
  },
  chip: {
    margin: theme.spacing(1),
  },
  DsName: {
    fontSize: theme.typography.fontSize,
    fontWeight: theme.typography.fontWeightNormal,
  }
})

class ListDataSetTable extends React.Component {
  static propTypes = {
    DatasetList: PropTypes.array,
    handleClickHistory: PropTypes.func,
  }

  state = {
    menuAnchor: null,
    currentDataset: "",
    currentBranch: ""
  }

  onShowChipMenu = (datasetName, branchName, e) => {
    this.setState({
      menuAnchor: e.target,
      currentDataset: datasetName,
      currentBranch: branchName
    })
  }

  onCloseChipMenu = () => {
    this.setState({
      menuAnchor: false,
      currentDataset: "",
      currentBranch: ""
    })
  }
  
  onShowEditMenu = (datasetName, e) => {
    this.setState({
      menuAnchor: e.target,
      currentDataset: datasetName
    })
  }

  onCloseEditMenu = () => {
    this.setState({
      menuAnchor: false,
      currentDataset: ""
    })
  }

  render() {
    const {
      classes,
      Datasets
    } = this.props

    return (
      <Fragment>
        <Table className={classes.table}>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Task</TableCell>
              <TableCell>Size</TableCell>
              <TableCell>Uploaded At</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
          {Datasets.map(x => {
            return (
              <TableRow key={x.id} hover>
                <TableCell>{x.id.slice(0,8)}</TableCell>
                <TableCell>{x.name}</TableCell>
                <TableCell>{x.task}</TableCell>
                <TableCell>{x.size_bytes} bytes</TableCell>
                <TableCell>{moment(x.datetime_created).fromNow()}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
        </Table>
       
      </Fragment>
    )
  }
}

export default withStyles(styles)(ListDataSetTable);
